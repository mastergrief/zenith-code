"""Live CPU torch tensor census with deduplicated unique storage accounting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def profile_torch_cpu_census_enabled() -> bool:
    import os

    rss_on = os.environ.get("HRM_TEXT_158_PROFILE_HOST_RSS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    census_on = os.environ.get("HRM_TEXT_158_PROFILE_TORCH_CPU_CENSUS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return rss_on and census_on


def torch_cpu_tensor_census(*, top_k: int = 10) -> dict[str, Any]:
    """Enumerate live CPU tensors without retaining tensor/storage objects (FOLD 3).

    Totals and dominance use GLOBAL UNIQUE backing storage keyed by
    (device, untyped_storage().data_ptr()) with storage.nbytes() (FOLD 1-2).
    """
    import gc

    import torch

    n_tensors = 0
    n_cpu_tensors = 0
    skipped_no_storage = 0
    skipped_non_cpu = 0
    logical_tensor_bytes = 0

    logical_groups: dict[tuple[str, str, tuple[int, ...]], dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "logical_tensor_bytes": 0}
    )
    storage_map: dict[tuple[str, int], dict[str, Any]] = {}

    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        n_tensors += 1
        try:
            device = str(obj.device)
            if device != "cpu":
                skipped_non_cpu += 1
                continue
            n_cpu_tensors += 1
            logical_bytes = int(obj.element_size() * obj.numel())
            logical_tensor_bytes += logical_bytes
            shape = tuple(int(dim) for dim in obj.shape)
            dtype = str(obj.dtype)
            group_key = (device, dtype, shape)
            logical_groups[group_key]["tensor_count"] += 1
            logical_groups[group_key]["logical_tensor_bytes"] += logical_bytes

            try:
                storage = obj.untyped_storage()
                data_ptr = int(storage.data_ptr())
                storage_nbytes = int(storage.nbytes())
            except Exception:
                skipped_no_storage += 1
                continue

            storage_key = (device, data_ptr)
            if storage_key not in storage_map:
                storage_map[storage_key] = {
                    "storage_nbytes": storage_nbytes,
                    "device": device,
                    "canonical_group": group_key,
                    "tensor_count": 0,
                    "alternate_groups": set(),
                }
            entry = storage_map[storage_key]
            entry["tensor_count"] += 1
            if group_key != entry["canonical_group"]:
                entry["alternate_groups"].add(group_key)
        except Exception:
            skipped_no_storage += 1
            continue

    group_unique_bytes: dict[tuple[str, str, tuple[int, ...]], int] = defaultdict(int)
    group_unique_storage_count: dict[tuple[str, str, tuple[int, ...]], int] = defaultdict(int)
    cross_group_shared_storage: list[dict[str, Any]] = []

    for storage_key, entry in storage_map.items():
        canonical = entry["canonical_group"]
        nbytes = int(entry["storage_nbytes"])
        group_unique_bytes[canonical] += nbytes
        group_unique_storage_count[canonical] += 1
        if entry["alternate_groups"]:
            cross_group_shared_storage.append(
                {
                    "storage_ptr": int(storage_key[1]),
                    "storage_nbytes": nbytes,
                    "canonical_group": {
                        "device": canonical[0],
                        "dtype": canonical[1],
                        "shape": list(canonical[2]),
                    },
                    "alternate_groups": [
                        {"device": g[0], "dtype": g[1], "shape": list(g[2])}
                        for g in sorted(entry["alternate_groups"])
                    ],
                }
            )

    unique_storage_bytes = sum(int(entry["storage_nbytes"]) for entry in storage_map.values())
    top_groups: list[dict[str, Any]] = []
    for group_key, unique_bytes in sorted(
        group_unique_bytes.items(),
        key=lambda item: int(item[1]),
        reverse=True,
    )[: int(top_k)]:
        device, dtype, shape = group_key
        logical = logical_groups[group_key]
        storage_count = int(group_unique_storage_count[group_key])
        top_groups.append(
            {
                "device": device,
                "dtype": dtype,
                "shape": list(shape),
                "tensor_count": int(logical["tensor_count"]),
                "logical_tensor_bytes": int(logical["logical_tensor_bytes"]),
                "unique_storage_bytes": int(unique_bytes),
                "unique_storage_count": storage_count,
                "bytes_per_storage_instance": (
                    int(unique_bytes // storage_count) if storage_count > 0 else int(unique_bytes)
                ),
            }
        )

    return {
        "n_tensors": int(n_tensors),
        "n_cpu_tensors": int(n_cpu_tensors),
        "n_unique_storages": int(len(storage_map)),
        "logical_tensor_bytes": int(logical_tensor_bytes),
        "unique_storage_bytes": int(unique_storage_bytes),
        "skipped_no_storage": int(skipped_no_storage),
        "skipped_non_cpu": int(skipped_non_cpu),
        "top_groups": top_groups,
        "cross_group_shared_storage": cross_group_shared_storage[:5],
    }
