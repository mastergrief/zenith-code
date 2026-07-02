"""Default-off C4 retention owner census (Slice 8m).

Tier A: len-only owner counts on obmalloc_C4_after_state allocation_dims.
Tier B: weakref-only live-object enrichment (never stores strong refs).
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import weakref
from typing import Any, Iterator, Mapping

PROFILE_C4_RETENTION_OWNER_CENSUS_ENV = "HRM_TEXT_158_PROFILE_C4_RETENTION_OWNER_CENSUS"

_c4_census_tls = threading.local()
_PENDING_OBMALLOC_AFTER_STATE_DIMS: dict[int, dict[str, int]] = {}
_APPEND_PATCH_INSTALLED = False


def profile_c4_retention_owner_census_enabled() -> bool:
    import os

    return os.environ.get(PROFILE_C4_RETENTION_OWNER_CENSUS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _estimate_nbytes(obj: Any) -> int:
    nbytes_fn = getattr(obj, "nbytes", None)
    if callable(nbytes_fn):
        try:
            return int(nbytes_fn())
        except Exception:
            return 0
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        try:
            return int(obj.numel()) * int(obj.element_size())
        except Exception:
            return 0
    return 0


class C4RetentionOwnerWeakrefRegistry:
    """Strong-ref-free owner registry for Tier B enrichment."""

    def __init__(self) -> None:
        self._refs: list[weakref.ref[Any]] = []
        self._live: dict[int, tuple[str, int]] = {}

    def register(self, obj: Any, *, owner_tag: str) -> None:
        obj_id = int(id(obj))

        def _on_dead(_ref: weakref.ref[Any], *, _obj_id: int = obj_id) -> None:
            self._live.pop(_obj_id, None)

        try:
            self._refs.append(weakref.ref(obj, _on_dead))
        except TypeError:
            return
        self._live[obj_id] = (str(owner_tag), _estimate_nbytes(obj))

    def live_counts_by_tag(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tag, _nbytes in self._live.values():
            counts[tag] = int(counts.get(tag, 0)) + 1
        return counts

    def live_nbytes_by_tag(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for tag, nbytes in self._live.values():
            totals[tag] = int(totals.get(tag, 0)) + int(nbytes)
        return totals

    def all_weakrefs_dead(self) -> bool:
        return not self._live


class C4RetentionOwnerCensusSession:
    def __init__(self) -> None:
        self._registry = C4RetentionOwnerWeakrefRegistry()
        self._registered_prior_ids: set[int] = set()

    def register_priors(
        self,
        tensor_states: Mapping[str, Any],
        event_states: Mapping[str, Any],
    ) -> None:
        for state_key, state in tensor_states.items():
            carrier = getattr(state, "event_coded_live_carrier", None)
            if carrier is not None:
                obj_id = id(carrier)
                if obj_id not in self._registered_prior_ids:
                    self._registry.register(carrier, owner_tag="prior_tensor_states")
                    self._registered_prior_ids.add(obj_id)
            vu = event_states.get(state_key)
            if vu is not None:
                carrier = getattr(vu, "carrier", None)
                if carrier is not None and id(carrier) not in self._registered_prior_ids:
                    self._registry.register(carrier, owner_tag="prior_event_states_alias")
                    self._registered_prior_ids.add(id(carrier))

    def register_new(self, *, carrier: Any, q_out: Any) -> None:
        self._registry.register(carrier, owner_tag="new_carriers_by_key")
        self._registry.register(q_out, owner_tag="new_q_by_key")

    @property
    def registry(self) -> C4RetentionOwnerWeakrefRegistry:
        return self._registry

    def tier_a_dims(
        self,
        *,
        tensor_states: Mapping[str, Any],
        event_states: Mapping[str, Any],
        carriers_by_key: Mapping[str, Any],
        q_by_key: Mapping[str, Any],
        next_states: Mapping[str, Any] | None,
        state_index: int,
    ) -> dict[str, int]:
        return {
            "c4_n_tensor_states": len(tensor_states),
            "c4_n_event_states": len(event_states),
            "c4_n_carriers_by_key": len(carriers_by_key),
            "c4_n_q_by_key": len(q_by_key),
            "c4_n_next_states": len(next_states or {}),
            "c4_state_index": int(state_index),
        }

    def tier_b_dims(self) -> dict[str, int]:
        counts = self._registry.live_counts_by_tag()
        return {
            "c4_weakref_n_prior_tensor_states": int(counts.get("prior_tensor_states", 0)),
            "c4_weakref_n_prior_event_states_alias": int(
                counts.get("prior_event_states_alias", 0)
            ),
            "c4_weakref_n_new_carriers_by_key": int(counts.get("new_carriers_by_key", 0)),
            "c4_weakref_n_new_q_by_key": int(counts.get("new_q_by_key", 0)),
        }

    def build_allocation_dims(
        self,
        *,
        tensor_states: Mapping[str, Any],
        event_states: Mapping[str, Any],
        carriers_by_key: Mapping[str, Any],
        q_by_key: Mapping[str, Any],
        next_states: Mapping[str, Any] | None,
        state_index: int,
    ) -> dict[str, int]:
        dims = self.tier_a_dims(
            tensor_states=tensor_states,
            event_states=event_states,
            carriers_by_key=carriers_by_key,
            q_by_key=q_by_key,
            next_states=next_states,
            state_index=state_index,
        )
        dims.update(self.tier_b_dims())
        return dims


def install_obmalloc_allocation_dims_append_patch() -> None:
    global _APPEND_PATCH_INSTALLED
    if _APPEND_PATCH_INSTALLED:
        return
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    original = probe._append_host_rss_profile_mark

    def _patched_append(path: Any, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event"))
        if event == "obmalloc_C4_after_state":
            state_index = payload.get("state_index")
            dims = None
            if state_index is not None:
                dims = _PENDING_OBMALLOC_AFTER_STATE_DIMS.pop(int(state_index), None)
            if dims is None:
                dims = getattr(_c4_census_tls, "allocation_dims", None)
            if dims is not None:
                merged = dict(payload)
                merged["allocation_dims"] = dict(dims)
                return original(path, merged)
        return original(path, payload)

    probe._append_host_rss_profile_mark = _patched_append  # type: ignore[assignment]
    _APPEND_PATCH_INSTALLED = True


@contextmanager
def pending_obmalloc_c4_after_state_allocation_dims(
    allocation_dims: Mapping[str, int] | None,
    *,
    state_index: int | None = None,
) -> Iterator[None]:
    if allocation_dims is None:
        yield
        return
    dims = dict(allocation_dims)
    if state_index is not None:
        _PENDING_OBMALLOC_AFTER_STATE_DIMS[int(state_index)] = dims
    prev = getattr(_c4_census_tls, "allocation_dims", None)
    _c4_census_tls.allocation_dims = dims
    try:
        yield
    finally:
        if state_index is not None:
            _PENDING_OBMALLOC_AFTER_STATE_DIMS.pop(int(state_index), None)
        if prev is None:
            if hasattr(_c4_census_tls, "allocation_dims"):
                delattr(_c4_census_tls, "allocation_dims")
        else:
            _c4_census_tls.allocation_dims = prev


def begin_c4_retention_owner_census_session() -> C4RetentionOwnerCensusSession | None:
    if not profile_c4_retention_owner_census_enabled():
        return None
    install_obmalloc_allocation_dims_append_patch()
    return C4RetentionOwnerCensusSession()


if profile_c4_retention_owner_census_enabled():
    install_obmalloc_allocation_dims_append_patch()
