"""R2-A-M1 CPU lossless equivalence: saved-tensor-hook seam remat (v5 mechanism)."""
from __future__ import annotations

import inspect
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from calm.hrm_text_158.native_full_stack.activation_relief import (
    ACTIVATION_RESIDUAL_TARGET_FAMILIES,
    ACTIVATION_RESIDUALS_FAIL_CLOSED_NON_CLAIMS,
    ZL_INIT_FP_EXCEPTION_CLASSIFICATION,
    ZL_INIT_FP_EXCEPTION_REGISTRY_ANCHOR,
    ZL_INIT_HRM_SOURCE_ANCHOR,
    build_activation_residuals_fail_closed_receipt,
    validate_activation_residuals_fail_closed_receipt,
    zL_init_observation_from_hrm_module,
)

MECHANISM_ID = "tier1_lossless_seam_saved_tensor_hook_remat_v5"
LOSSLESS_EQUIV_SCHEMA_VERSION = (
    "hrm_text_158_activation_residuals_lossless_equiv/v6.cpu_saved_tensor_hook_remat_v5"
)
LOSSLESS_EQUIV_TARGET_NAME = "r2a_activation_residuals_m1_lossless_equivalence"
PROOF_KIND_CPU_PRODUCTION_LOSSLESS_EQUIVALENCE = "cpu_production_lossless_equivalence"
SAVED_TENSOR_HOOK_SOURCE = "torch.autograd.graph.saved_tensors_hooks"

SANITIZE_STRIP_KEYS = (
    "activation_codec_seam",
    "activation_relief_policy",
    "remat_instrumentation",
)

LOSSLESS_EQUIV_NON_CLAIMS = (
    "proves CPU Tier1 saved-tensor-hook seam remat with closure-audited no-hidden authority for four families only",
    "does not authorize activations_residuals row flip on live scaffold",
    "does not substitute for GPU memory measurement (REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS)",
    "does not claim learning, acquisition, retention, throughput, or .pt mutation",
    "zL_init persistent FP-exception non-claim remains outside the four-family allowlist",
    "graph-resident producing-block inputs in closures are allowed; GPU peak-memory relief is R2-A-L gate (c)",
)


def sanitize_seq_info_for_recompute(seq_info: Mapping[str, Any] | None) -> dict[str, Any]:
    if not seq_info:
        return {}
    return {k: v for k, v in dict(seq_info).items() if k not in SANITIZE_STRIP_KEYS}


@dataclass(frozen=True)
class UpstreamRecomputeKeyBinding:
    recompute_key: int
    source_tensor_id: int
    binding_class: str = "upstream_recompute_key"


@dataclass(frozen=True)
class GraphResidentInputBinding:
    tensor: Tensor
    shape: tuple[int, ...]
    dtype: str
    device: str
    binding_class: str = "graph_resident_input"


@dataclass(frozen=True)
class AuxiliaryGraphInputBinding:
    role: str
    tensor: Tensor
    shape: tuple[int, ...]
    dtype: str
    device: str
    binding_class: str = "auxiliary_graph_input"


@dataclass(frozen=True)
class ModelOwnedBinding:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    binding_class: str = "model_owned"


CaptureBinding = (
    UpstreamRecomputeKeyBinding
    | GraphResidentInputBinding
    | AuxiliaryGraphInputBinding
    | ModelOwnedBinding
)


def resolve_capture_binding(
    tensor: Tensor,
    *,
    handle_registry: SeamTensorHandleRegistry,
    model: torch.nn.Module,
    aux_role_by_id: Mapping[int, str],
) -> CaptureBinding:
    key = handle_registry.lookup_registered_seam_tensor(tensor)
    if key is not None:
        return UpstreamRecomputeKeyBinding(
            recompute_key=key,
            source_tensor_id=id(tensor),
        )
    tensor_id = id(tensor)
    if tensor_id in aux_role_by_id:
        return AuxiliaryGraphInputBinding(
            role=aux_role_by_id[tensor_id],
            tensor=tensor,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )
    for name, param in model.named_parameters():
        if param is tensor:
            return ModelOwnedBinding(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )
    for name, buf in model.named_buffers():
        if buf is tensor:
            return ModelOwnedBinding(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )
    return GraphResidentInputBinding(
        tensor=tensor,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def materialize_binding(
    binding: CaptureBinding,
    *,
    recipe_store: SeamRematRecipeStore,
    model: torch.nn.Module,
) -> Tensor:
    if isinstance(binding, UpstreamRecomputeKeyBinding):
        return recipe_store.recompute(binding.recompute_key)
    if isinstance(binding, ModelOwnedBinding):
        for name, param in model.named_parameters():
            if name == binding.name:
                return param
        for name, buf in model.named_buffers():
            if name == binding.name:
                return buf
        raise ValueError(f"model-owned binding {binding.name!r} not found")
    if isinstance(binding, (GraphResidentInputBinding, AuxiliaryGraphInputBinding)):
        return binding.tensor
    raise TypeError(f"unsupported binding: {type(binding)}")


@dataclass
class SeamTensorHandleRegistry:
    _tensor_id_to_key: dict[int, int] = field(default_factory=dict)
    _key_to_tensor_ref: dict[int, weakref.ReferenceType[Tensor]] = field(
        default_factory=dict
    )

    def clear(self) -> None:
        self._tensor_id_to_key.clear()
        self._key_to_tensor_ref.clear()

    def register_seam_tensor(self, tensor: Tensor, recompute_key: int) -> None:
        self._tensor_id_to_key[id(tensor)] = recompute_key
        self._key_to_tensor_ref[recompute_key] = weakref.ref(tensor)

    def is_registered_seam_tensor_or_alias(self, tensor: Tensor) -> bool:
        for ref in self._key_to_tensor_ref.values():
            registered = ref()
            if registered is None:
                continue
            if registered is tensor:
                return True
            if (
                registered.data_ptr() == tensor.data_ptr()
                and registered.shape == tensor.shape
                and registered.dtype == tensor.dtype
            ):
                return True
        return False

    def lookup_registered_seam_tensor(self, tensor: Tensor) -> int | None:
        key = self._tensor_id_to_key.get(id(tensor))
        if key is None:
            return None
        ref = self._key_to_tensor_ref.get(key)
        if ref is None or ref() is not tensor:
            return None
        return key


@dataclass
class RecomputeFnEntry:
    family: str
    recompute_key: int
    recompute_fn: Callable[[], Tensor]
    upstream_recompute_keys: tuple[int, ...]
    closure_capture_audit: dict[str, object]


@dataclass
class SeamRematRecipeStore:
    _entries: dict[int, RecomputeFnEntry] = field(default_factory=dict)
    _next_key: int = 1
    _recomputing: set[int] = field(default_factory=set)
    pack_events: list[dict[str, object]] = field(default_factory=list)
    unpack_recompute_count: int = 0
    registration_count: int = 0

    def clear(self) -> None:
        self._entries.clear()
        self._next_key = 1
        self._recomputing.clear()
        self.pack_events.clear()
        self.unpack_recompute_count = 0
        self.registration_count = 0

    def allocate_key(self) -> int:
        key = self._next_key
        self._next_key += 1
        return key

    def register_entry(self, entry: RecomputeFnEntry) -> None:
        self._entries[entry.recompute_key] = entry
        self.registration_count += 1

    def record_pack(
        self,
        *,
        recompute_key: int,
        tensor_id: int,
        kind: str = "handle",
    ) -> None:
        self.pack_events.append(
            {"kind": kind, "recompute_key": recompute_key, "tensor_id": tensor_id}
        )

    def record_full_tensor_pack(self, *, tensor_id: int) -> None:
        self.pack_events.append({"kind": "full_tensor", "tensor_id": tensor_id})

    def recompute(self, recompute_key: int) -> Tensor:
        if recompute_key in self._recomputing:
            raise ValueError(f"cycle detected in seam remat recompute at key={recompute_key}")
        entry = self._entries.get(recompute_key)
        if entry is None:
            raise ValueError(f"unknown recompute_key={recompute_key}")
        self._recomputing.add(recompute_key)
        try:
            for upstream_key in entry.upstream_recompute_keys:
                self.recompute(upstream_key)
            self.unpack_recompute_count += 1
            return entry.recompute_fn()
        finally:
            self._recomputing.discard(recompute_key)


def audit_recompute_closure(
    recompute_fn: Callable[[], Tensor],
    *,
    handle_registry: SeamTensorHandleRegistry,
) -> dict[str, object]:
    closure_vars = inspect.getclosurevars(recompute_fn)
    forbidden_closure_tensor_count = 0
    registered_seam_tensor_in_closure_count = 0
    graph_resident_input_count = 0
    upstream_key_count = 0
    auxiliary_count = 0
    model_owned_count = 0
    reachable_tensors: list[dict[str, object]] = []

    def _inspect_obj(obj: object) -> None:
        nonlocal forbidden_closure_tensor_count
        nonlocal registered_seam_tensor_in_closure_count
        nonlocal graph_resident_input_count
        nonlocal upstream_key_count
        nonlocal auxiliary_count
        nonlocal model_owned_count
        if isinstance(obj, dict):
            for value in obj.values():
                _inspect_obj(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                _inspect_obj(value)
            return
        if isinstance(obj, CaptureBinding):
            if isinstance(obj, UpstreamRecomputeKeyBinding):
                upstream_key_count += 1
            elif isinstance(obj, GraphResidentInputBinding):
                graph_resident_input_count += 1
                _inspect_obj(obj.tensor)
            elif isinstance(obj, AuxiliaryGraphInputBinding):
                auxiliary_count += 1
            elif isinstance(obj, ModelOwnedBinding):
                model_owned_count += 1
            return
        if isinstance(obj, Tensor):
            key = handle_registry.lookup_registered_seam_tensor(obj)
            reachable_tensors.append(
                {
                    "tensor_id": id(obj),
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "registered": key is not None,
                }
            )
            if key is not None:
                forbidden_closure_tensor_count += 1
                registered_seam_tensor_in_closure_count += 1

    capture_meta = getattr(recompute_fn, "__closure_capture__", None)
    if capture_meta is not None:
        _inspect_obj(capture_meta)

    for binding in closure_vars.nonlocals.values():
        _inspect_obj(binding)
    for val in closure_vars.globals.values():
        _inspect_obj(val)

    return {
        "forbidden_closure_tensor_count": forbidden_closure_tensor_count,
        "registered_seam_tensor_in_closure_count": registered_seam_tensor_in_closure_count,
        "seam_output_alias_or_clone_in_closure": registered_seam_tensor_in_closure_count > 0,
        "graph_resident_input_count": graph_resident_input_count,
        "upstream_recompute_key_count": upstream_key_count,
        "auxiliary_graph_input_count": auxiliary_count,
        "model_owned_tensor_count": model_owned_count,
        "reachable_tensors": tuple(reachable_tensors),
    }


class Tier1LosslessSeamSavedTensorHookRematCodec:
    """Production-path seam codec with universal capture-to-key remat."""

    def __init__(self, *, model: torch.nn.Module) -> None:
        self.model = model
        self.handle_registry = SeamTensorHandleRegistry()
        self.recipe_store = SeamRematRecipeStore()
        self._in_probe = False
        self._registration_frozen = False
        self._hook_scope_active = False
        self._hook_scope_wraps_forward_saves = False
        self.seam_events: list[dict[str, object]] = []
        self.per_family_records: list[dict[str, object]] = []
        self.recompute_registration_side_effect_count = 0
        self.recompute_seam_callback_invocation_count = 0
        self.pre_unpack_probe_saved_pack_count = 0
        self.pre_unpack_probe_registration_count = 0
        self._registration_count_at_scope_start = 0
        self._aux_role_by_id: dict[int, str] = {}

    def set_auxiliary_tensors(self, **named: Tensor) -> None:
        self._aux_role_by_id = {id(tensor): role for role, tensor in named.items()}

    @contextmanager
    def probe_context(self):
        prev = self._in_probe
        prev_frozen = self._registration_frozen
        reg_before = self.recipe_store.registration_count
        pack_before = len(self.recipe_store.pack_events)
        self._in_probe = True
        self._registration_frozen = True
        try:
            yield
        finally:
            self._in_probe = prev
            self._registration_frozen = prev_frozen
            self.pre_unpack_probe_registration_count += max(
                0,
                self.recipe_store.registration_count - reg_before,
            )
            self.pre_unpack_probe_saved_pack_count += max(
                0,
                len(self.recipe_store.pack_events) - pack_before,
            )

    @contextmanager
    def saved_tensor_hook_scope(self):
        self.handle_registry.clear()
        self.recipe_store.clear()
        self.per_family_records.clear()
        self._registration_count_at_scope_start = 0

        def pack_hook(tensor: Tensor) -> object:
            key = self.handle_registry.lookup_registered_seam_tensor(tensor)
            if key is not None:
                self.recipe_store.record_pack(
                    recompute_key=key,
                    tensor_id=id(tensor),
                    kind="handle",
                )
                return ("seam_remat_handle", key)
            if self.handle_registry.is_registered_seam_tensor_or_alias(tensor):
                self.recipe_store.record_full_tensor_pack(tensor_id=id(tensor))
            return tensor

        def unpack_hook(saved: object) -> Tensor:
            if isinstance(saved, tuple) and saved[0] == "seam_remat_handle":
                return self.recipe_store.recompute(int(saved[1]))
            return saved  # type: ignore[return-value]

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            self._hook_scope_active = True
            try:
                yield
                self._hook_scope_wraps_forward_saves = True
            finally:
                self._hook_scope_active = False

    def _resolve_binding(self, tensor: Tensor) -> CaptureBinding:
        return resolve_capture_binding(
            tensor,
            handle_registry=self.handle_registry,
            model=self.model,
            aux_role_by_id=self._aux_role_by_id,
        )

    def _bindings_from_producing_inputs(
        self,
        producing_inputs: Sequence[Tensor],
    ) -> tuple[CaptureBinding, ...]:
        return tuple(self._resolve_binding(tensor) for tensor in producing_inputs)

    def _bindings_from_optional_tensor(
        self,
        tensor: Tensor | None,
    ) -> Tensor | None:
        if tensor is None:
            return None
        return materialize_binding(
            self._resolve_binding(tensor),
            recipe_store=self.recipe_store,
            model=self.model,
        )

    def _bindings_from_cos_sin(self, cos_sin: object | None) -> object | None:
        if cos_sin is None:
            return None
        if isinstance(cos_sin, tuple):
            bound: list[Tensor | None] = []
            for item in cos_sin:
                if isinstance(item, Tensor):
                    bound.append(
                        materialize_binding(
                            self._resolve_binding(item),
                            recipe_store=self.recipe_store,
                            model=self.model,
                        )
                    )
                else:
                    bound.append(item)
            return tuple(bound)
        if isinstance(cos_sin, Tensor):
            return materialize_binding(
                self._resolve_binding(cos_sin),
                recipe_store=self.recipe_store,
                model=self.model,
            )
        return cos_sin

    def _bindings_from_sanitized_kwargs(
        self,
        sanitized: Mapping[str, Any],
    ) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        for key, value in sanitized.items():
            if isinstance(value, Tensor):
                bound[key] = materialize_binding(
                    self._resolve_binding(value),
                    recipe_store=self.recipe_store,
                    model=self.model,
                )
            elif isinstance(value, dict):
                bound[key] = self._bindings_from_sanitized_kwargs(value)
            elif isinstance(value, (list, tuple)):
                bound[key] = type(value)(
                    materialize_binding(self._resolve_binding(item), recipe_store=self.recipe_store, model=self.model)
                    if isinstance(item, Tensor)
                    else item
                    for item in value
                )
            else:
                bound[key] = value
        return bound

    def _upstream_keys_from_bindings(
        self,
        bindings: Sequence[CaptureBinding],
    ) -> tuple[int, ...]:
        keys: list[int] = []
        for binding in bindings:
            if isinstance(binding, UpstreamRecomputeKeyBinding):
                keys.append(binding.recompute_key)
        return tuple(keys)

    def _build_recompute_fn(
        self,
        *,
        family: str,
        block: str,
        module: torch.nn.Module,
        bindings: Sequence[CaptureBinding],
        seq_kwargs: Mapping[str, Any] | None,
        cos_sin: object | None,
        sep_positions: Tensor | None,
    ) -> Callable[[], Tensor]:
        sanitized_raw = sanitize_seq_info_for_recompute(seq_kwargs)
        store = self.recipe_store
        model = self.model
        bound_sep = self._bindings_from_optional_tensor(sep_positions)
        bound_cos_sin = self._bindings_from_cos_sin(cos_sin)
        bound_sanitized = self._bindings_from_sanitized_kwargs(sanitized_raw)
        closure_capture: dict[str, CaptureBinding | Mapping[str, Any] | object] = {
            "sanitized_kwargs": bound_sanitized,
        }
        if sep_positions is not None:
            closure_capture["sep_positions"] = self._resolve_binding(sep_positions)
        if cos_sin is not None:
            if isinstance(cos_sin, tuple):
                closure_capture["cos_sin"] = tuple(
                    self._resolve_binding(item) if isinstance(item, Tensor) else item
                    for item in cos_sin
                )
            elif isinstance(cos_sin, Tensor):
                closure_capture["cos_sin"] = self._resolve_binding(cos_sin)
            else:
                closure_capture["cos_sin"] = cos_sin

        def mat(binding: CaptureBinding) -> Tensor:
            value = materialize_binding(binding, recipe_store=store, model=model)
            if self._in_probe and isinstance(
                binding, (GraphResidentInputBinding, AuxiliaryGraphInputBinding)
            ):
                return value.detach().clone()
            return value

        if family == "recurrent.z_L_update":
            z_L_in, z_H_in = bindings
            closure_capture["z_L_in"] = z_L_in
            closure_capture["z_H_in"] = z_H_in

            def recompute_fn() -> Tensor:
                return module.L_level(
                    mat(z_L_in),
                    mat(z_H_in),
                    **bound_sanitized,
                )

            recompute_fn.__closure_capture__ = closure_capture  # type: ignore[attr-defined]
            return recompute_fn
        if family == "recurrent.z_H_update":
            z_H_in, z_L_in = bindings
            closure_capture["z_H_in"] = z_H_in
            closure_capture["z_L_in"] = z_L_in

            def recompute_fn() -> Tensor:
                return module.H_level(
                    mat(z_H_in),
                    mat(z_L_in),
                    **bound_sanitized,
                )

            recompute_fn.__closure_capture__ = closure_capture  # type: ignore[attr-defined]
            return recompute_fn
        if family == "residual.post_attn":
            (x_prev_binding,) = bindings
            closure_capture["x_prev"] = x_prev_binding
            norm_type = getattr(module, "_norm_type", "pre")

            def recompute_fn() -> Tensor:
                x_prev = mat(x_prev_binding)
                if norm_type == "pre":
                    return x_prev + module.attn(
                        module.norm(x_prev),
                        cos_sin=bound_cos_sin,
                        sep_positions=bound_sep,
                        **bound_sanitized,
                    )
                return x_prev + module.attn(
                    x_prev,
                    cos_sin=bound_cos_sin,
                    sep_positions=bound_sep,
                    **bound_sanitized,
                )

            recompute_fn.__closure_capture__ = closure_capture  # type: ignore[attr-defined]
            return recompute_fn
        if family == "residual.post_mlp":
            (x_aa_binding,) = bindings
            closure_capture["x_after_attn"] = x_aa_binding
            norm_type = getattr(module, "_norm_type", "pre")

            def recompute_fn() -> Tensor:
                x_aa = mat(x_aa_binding)
                if norm_type == "pre":
                    return x_aa + module.mlp(module.norm(x_aa))
                x_normed = module.norm(x_aa)
                return x_normed + module.mlp(x_normed)

            recompute_fn.__closure_capture__ = closure_capture  # type: ignore[attr-defined]
            return recompute_fn
        raise ValueError(f"unsupported remat family/block: {family}/{block}")

    def __call__(
        self,
        family: str,
        tensor: object,
        **ctx: Any,
    ) -> object:
        if family not in ACTIVATION_RESIDUAL_TARGET_FAMILIES:
            return tensor
        if not isinstance(tensor, Tensor):
            raise ValueError(f"activation_codec_seam expected torch.Tensor for {family!r}")
        # Nested seam hits during pre-unpack probe recompute must not recurse probe/register.
        if self._registration_frozen:
            return tensor

        producing_inputs = tuple(ctx.get("producing_inputs", ()))
        seq_kwargs = ctx.get("seq_kwargs")
        block = str(ctx.get("block", ""))
        module = ctx.get("module", self.model)
        cos_sin = ctx.get("cos_sin")
        sep_positions = ctx.get("sep_positions")

        bindings = self._bindings_from_producing_inputs(producing_inputs)
        recompute_fn = self._build_recompute_fn(
            family=family,
            block=block,
            module=module,
            bindings=bindings,
            seq_kwargs=seq_kwargs,
            cos_sin=cos_sin,
            sep_positions=sep_positions,
        )

        with torch.no_grad():
            with self.probe_context():
                actual = recompute_fn()
            torch.testing.assert_close(actual, tensor, atol=0.0, rtol=0.0)

        audit = audit_recompute_closure(
            recompute_fn,
            handle_registry=self.handle_registry,
        )
        recompute_key = self.recipe_store.allocate_key()
        upstream_keys = self._upstream_keys_from_bindings(bindings)
        entry = RecomputeFnEntry(
            family=family,
            recompute_key=recompute_key,
            recompute_fn=recompute_fn,
            upstream_recompute_keys=upstream_keys,
            closure_capture_audit=audit,
        )
        if not self._registration_frozen:
            self.recipe_store.register_entry(entry)
            seam_output = tensor.clone()
            self.handle_registry.register_seam_tensor(seam_output, recompute_key)
        else:
            seam_output = tensor

        self.seam_events.append(
            {
                "family": family,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": bool(tensor.requires_grad),
                "mechanism": "tier1_lossless_seam_saved_tensor_hook_remat",
                "recompute_key": recompute_key,
                "upstream_recompute_keys": list(upstream_keys),
                "binding_classes": [b.binding_class for b in bindings],
            }
        )
        self.per_family_records.append(
            {
                "family": family,
                "event_count": 1,
                "pre_unpack_recompute_exact": True,
                "closure_capture_audit": audit,
                "upstream_recompute_keys": list(upstream_keys),
            }
        )
        return seam_output

    def telemetry(self) -> dict[str, object]:
        handle_pack_count = len(self.recipe_store.pack_events)
        full_pack_count = sum(
            1
            for event in self.recipe_store.pack_events
            if event.get("kind") == "full_tensor"
        )
        audits = [r["closure_capture_audit"] for r in self.per_family_records]
        forbidden = sum(int(a["forbidden_closure_tensor_count"]) for a in audits)
        registered_in_closure = sum(
            int(a["registered_seam_tensor_in_closure_count"]) for a in audits
        )
        return {
            "mechanism_id": MECHANISM_ID,
            "hook_scope_wraps_forward_saves": self._hook_scope_wraps_forward_saves,
            "registered_seam_tensor_pack_count": handle_pack_count,
            "seam_handle_pack_count": handle_pack_count,
            "registered_seam_tensor_full_pack_count": full_pack_count,
            "m1_seam_full_tensor_save_count_at_pack": full_pack_count,
            "m1_seam_remat_unpack_recompute_count_total": self.recipe_store.unpack_recompute_count,
            "recompute_registration_side_effect_count": self.recompute_registration_side_effect_count,
            "recompute_seam_callback_invocation_count": self.recompute_seam_callback_invocation_count,
            "pre_unpack_probe_saved_pack_count": self.pre_unpack_probe_saved_pack_count,
            "pre_unpack_probe_registration_count": self.pre_unpack_probe_registration_count,
            "universal_capture_to_key_rule_applied": True,
            "cross_family_full_seam_output_tensor_capture_count": registered_in_closure,
            "registered_seam_tensor_in_closure_count": registered_in_closure,
            "forbidden_closure_tensor_count_total": forbidden,
            "sanitized_kwargs_strip_keys_applied": list(SANITIZE_STRIP_KEYS),
            "per_family_seam_records": tuple(self.per_family_records),
            "seam_events": tuple(self.seam_events),
        }


def build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(
    *,
    model: torch.nn.Module,
) -> Tier1LosslessSeamSavedTensorHookRematCodec:
    return Tier1LosslessSeamSavedTensorHookRematCodec(model=model)


@dataclass(frozen=True)
class TrainerActivationResidualsLosslessEquivalenceReceipt:
    schema_version: str
    target_name: str
    proof_kind: str
    mechanism_id: str
    source_commit_sha: str
    proof_command_argv: tuple[str, ...]
    main_path_proven: bool
    main_autograd_path_differs_from_baseline: bool
    universal_capture_to_key_rule_applied: bool
    hook_scope_wraps_forward_saves: bool
    registered_seam_tensor_pack_count: int
    seam_handle_pack_count: int
    registered_seam_tensor_full_pack_count: int
    m1_seam_remat_unpack_recompute_count_total: int
    registered_seam_tensor_in_closure_count: int
    cross_family_full_seam_output_tensor_capture_count: int
    recompute_registration_side_effect_count: int
    fail_closed_receipt: Any
    live_readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    optimizer_step_called: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "proof_kind": self.proof_kind,
            "mechanism_id": self.mechanism_id,
            "source_commit_sha": self.source_commit_sha,
            "proof_command_argv": list(self.proof_command_argv),
            "main_path_proven": self.main_path_proven,
            "main_autograd_path_differs_from_baseline": (
                self.main_autograd_path_differs_from_baseline
            ),
            "universal_capture_to_key_rule_applied": (
                self.universal_capture_to_key_rule_applied
            ),
            "hook_scope_wraps_forward_saves": self.hook_scope_wraps_forward_saves,
            "registered_seam_tensor_pack_count": self.registered_seam_tensor_pack_count,
            "seam_handle_pack_count": self.seam_handle_pack_count,
            "registered_seam_tensor_full_pack_count": (
                self.registered_seam_tensor_full_pack_count
            ),
            "m1_seam_remat_unpack_recompute_count_total": (
                self.m1_seam_remat_unpack_recompute_count_total
            ),
            "registered_seam_tensor_in_closure_count": (
                self.registered_seam_tensor_in_closure_count
            ),
            "cross_family_full_seam_output_tensor_capture_count": (
                self.cross_family_full_seam_output_tensor_capture_count
            ),
            "recompute_registration_side_effect_count": (
                self.recompute_registration_side_effect_count
            ),
            "fail_closed_receipt": self.fail_closed_receipt.to_dict(),
            "live_readiness_row_flip_authorized": self.live_readiness_row_flip_authorized,
            "readiness_row_flip_authorized_surface_names": list(
                self.readiness_row_flip_authorized_surface_names
            ),
            "optimizer_step_called": self.optimizer_step_called,
            "non_claims": list(self.non_claims),
        }


def build_trainer_activation_residuals_lossless_equivalence_receipt(
    *,
    source_commit_sha: str,
    proof_command_argv: Sequence[str],
    seam_events: Sequence[Mapping[str, object]],
    zL_init_observation: Mapping[str, object],
    telemetry: Mapping[str, object],
    main_path_proven: bool,
    main_autograd_path_differs_from_baseline: bool,
) -> TrainerActivationResidualsLosslessEquivalenceReceipt:
    unpack_total = int(telemetry["m1_seam_remat_unpack_recompute_count_total"])
    registered_in_closure = int(telemetry["registered_seam_tensor_in_closure_count"])
    mechanism_present = unpack_total > 0 and int(telemetry["seam_handle_pack_count"]) > 0
    no_hidden = (
        registered_in_closure == 0
        and int(telemetry["forbidden_closure_tensor_count_total"]) == 0
        and int(telemetry["recompute_registration_side_effect_count"]) == 0
    )
    fail_closed = build_activation_residuals_fail_closed_receipt(
        seam_events=seam_events,
        zL_init_observation=zL_init_observation,
        real_sub2_or_remat_or_offload_mechanism_present=mechanism_present,
        no_hidden_bf16_authority_proven=no_hidden,
        gpu_memory_receipt_present=False,
        ready_to_flip=False,
        activations_residuals_sub2_claim=False,
    )
    receipt = TrainerActivationResidualsLosslessEquivalenceReceipt(
        schema_version=LOSSLESS_EQUIV_SCHEMA_VERSION,
        target_name=LOSSLESS_EQUIV_TARGET_NAME,
        proof_kind=PROOF_KIND_CPU_PRODUCTION_LOSSLESS_EQUIVALENCE,
        mechanism_id=MECHANISM_ID,
        source_commit_sha=source_commit_sha,
        proof_command_argv=tuple(proof_command_argv),
        main_path_proven=bool(main_path_proven),
        main_autograd_path_differs_from_baseline=bool(
            main_autograd_path_differs_from_baseline
        ),
        universal_capture_to_key_rule_applied=True,
        hook_scope_wraps_forward_saves=bool(telemetry["hook_scope_wraps_forward_saves"]),
        registered_seam_tensor_pack_count=int(telemetry["registered_seam_tensor_pack_count"]),
        seam_handle_pack_count=int(telemetry["seam_handle_pack_count"]),
        registered_seam_tensor_full_pack_count=int(
            telemetry["registered_seam_tensor_full_pack_count"]
        ),
        m1_seam_remat_unpack_recompute_count_total=unpack_total,
        registered_seam_tensor_in_closure_count=registered_in_closure,
        cross_family_full_seam_output_tensor_capture_count=int(
            telemetry["cross_family_full_seam_output_tensor_capture_count"]
        ),
        recompute_registration_side_effect_count=int(
            telemetry["recompute_registration_side_effect_count"]
        ),
        fail_closed_receipt=fail_closed,
        live_readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        optimizer_step_called=False,
        non_claims=LOSSLESS_EQUIV_NON_CLAIMS,
    )
    validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)
    return receipt


def validate_trainer_activation_residuals_lossless_equivalence_receipt(
    receipt: TrainerActivationResidualsLosslessEquivalenceReceipt,
) -> None:
    if receipt.schema_version != LOSSLESS_EQUIV_SCHEMA_VERSION:
        raise ValueError("lossless equivalence receipt schema mismatch")
    if receipt.target_name != LOSSLESS_EQUIV_TARGET_NAME:
        raise ValueError("lossless equivalence receipt target mismatch")
    if receipt.proof_kind != PROOF_KIND_CPU_PRODUCTION_LOSSLESS_EQUIVALENCE:
        raise ValueError("lossless equivalence receipt proof_kind mismatch")
    if receipt.mechanism_id != MECHANISM_ID:
        raise ValueError("lossless equivalence receipt mechanism_id mismatch")
    if not receipt.universal_capture_to_key_rule_applied:
        raise ValueError("universal capture-to-key rule must be applied")
    if receipt.registered_seam_tensor_in_closure_count != 0:
        raise ValueError("registered seam tensors must not appear in closures")
    if receipt.cross_family_full_seam_output_tensor_capture_count != 0:
        raise ValueError("cross-family full seam tensor captures must be zero")
    if not receipt.main_path_proven:
        raise ValueError("main path must be proven")
    if not receipt.main_autograd_path_differs_from_baseline:
        raise ValueError("main autograd path must differ from baseline")
    if not receipt.hook_scope_wraps_forward_saves:
        raise ValueError("hook scope must wrap forward saves")
    if receipt.registered_seam_tensor_pack_count <= 0:
        raise ValueError("registered seam tensor pack count must be positive")
    if receipt.seam_handle_pack_count <= 0:
        raise ValueError("seam handle pack count must be positive")
    if receipt.m1_seam_remat_unpack_recompute_count_total <= 0:
        raise ValueError("unpack recompute count must be positive")
    if receipt.registered_seam_tensor_pack_count != receipt.seam_handle_pack_count:
        raise ValueError("handle pack count must match registered seam pack count")
    if receipt.registered_seam_tensor_full_pack_count != 0:
        raise ValueError("registered seam full pack count must be zero")
    if receipt.recompute_registration_side_effect_count != 0:
        raise ValueError("recompute registration side effects must be zero")
    validate_activation_residuals_fail_closed_receipt(receipt.fail_closed_receipt)
    fc = receipt.fail_closed_receipt
    if fc.ready_to_flip:
        raise ValueError("CPU lossless equivalence receipt cannot set ready_to_flip")
    if fc.gpu_memory_receipt_present:
        raise ValueError("CPU lossless equivalence receipt cannot claim GPU memory receipt")
    if fc.activations_residuals_sub2_claim:
        raise ValueError("CPU lossless equivalence receipt cannot claim sub2")
    if not fc.real_sub2_or_remat_or_offload_mechanism_present:
        raise ValueError("gate (a) requires active remat mechanism")
    if not fc.no_hidden_bf16_authority_proven:
        raise ValueError("gate (b) requires no-hidden audit pass")
    if receipt.live_readiness_row_flip_authorized:
        raise ValueError("CPU lossless equivalence receipt cannot authorize row flip")
    if receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("authorized surface list must be empty")
    if receipt.optimizer_step_called:
        raise ValueError("optimizer step must not be called")
    if receipt.non_claims != LOSSLESS_EQUIV_NON_CLAIMS:
        raise ValueError("lossless equivalence non-claims must be exact")


def gate_b_passes_from_audit(audit: Mapping[str, object]) -> bool:
    return (
        int(audit["forbidden_closure_tensor_count"]) == 0
        and int(audit["registered_seam_tensor_in_closure_count"]) == 0
    )
