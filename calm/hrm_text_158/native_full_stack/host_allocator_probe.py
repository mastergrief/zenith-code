"""Host allocator / smaps probe for non-torch RSS attribution (compact scalars only)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_VMA_HEADER_RE = re.compile(
    r"^[0-9a-f]+-[0-9a-f]+\s+[rwxsp\-]{4}\s+",
    re.IGNORECASE,
)


def _is_vma_header_line(line: str) -> bool:
    return bool(_VMA_HEADER_RE.match(line))


def profile_allocator_native_enabled() -> bool:
    import os

    rss_on = os.environ.get("HRM_TEXT_158_PROFILE_HOST_RSS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    alloc_on = os.environ.get("HRM_TEXT_158_PROFILE_ALLOCATOR_NATIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return rss_on and alloc_on


def profile_allocator_host_cache_diag_enabled() -> bool:
    import os

    return profile_allocator_native_enabled() and os.environ.get(
        "HRM_TEXT_158_PROFILE_ALLOCATOR_HOST_CACHE_DIAG",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _proc_kib_field(path: Path, key_prefix: str) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(key_prefix):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except Exception:
        return None
    return None


def read_smaps_rollup() -> dict[str, Any]:
    rollup_path = Path("/proc/self/smaps_rollup")
    out: dict[str, Any] = {}
    try:
        for key, field in (
            ("pss_kb", "Pss:"),
            ("rss_kb", "Rss:"),
            ("anonymous_kb", "Anonymous:"),
            ("private_dirty_kb", "Private_Dirty:"),
            ("private_clean_kb", "Private_Clean:"),
            ("shared_clean_kb", "Shared_Clean:"),
            ("shared_dirty_kb", "Shared_Dirty:"),
        ):
            value = _proc_kib_field(rollup_path, field)
            if value is not None:
                out[key] = int(value)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def read_smaps_categories() -> dict[str, Any]:
    """Compact category totals from /proc/self/smaps (no raw dump)."""
    categories = {
        "anonymous_kb": 0,
        "heap_kb": 0,
        "stack_kb": 0,
        "file_mapped_kb": 0,
        "shared_kb": 0,
        "other_kb": 0,
    }
    try:
        current_name = ""
        current_size_kb = 0
        for line in Path("/proc/self/smaps").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if _is_vma_header_line(line):
                if current_name and current_size_kb > 0:
                    name_lower = current_name.lower()
                    if "[heap]" in name_lower:
                        categories["heap_kb"] += current_size_kb
                    elif "[stack" in name_lower:
                        categories["stack_kb"] += current_size_kb
                    elif current_name == "[anon]" or "anon" in name_lower:
                        categories["anonymous_kb"] += current_size_kb
                    elif current_name.startswith("/") or ".so" in current_name:
                        categories["file_mapped_kb"] += current_size_kb
                    else:
                        categories["other_kb"] += current_size_kb
                parts = line.split()
                current_name = parts[-1] if len(parts) >= 6 else line.strip()
                current_size_kb = 0
                continue
            if line.startswith("Size:"):
                parts = line.split()
                if len(parts) >= 2:
                    current_size_kb = int(parts[1])
            elif line.startswith("Shared_Clean:") or line.startswith("Shared_Dirty:"):
                parts = line.split()
                if len(parts) >= 2:
                    categories["shared_kb"] += int(parts[1])
        if current_name and current_size_kb > 0:
            name_lower = current_name.lower()
            if "[heap]" in name_lower:
                categories["heap_kb"] += current_size_kb
            elif "[stack" in name_lower:
                categories["stack_kb"] += current_size_kb
            elif current_name == "[anon]" or "anon" in name_lower:
                categories["anonymous_kb"] += current_size_kb
            elif current_name.startswith("/") or ".so" in current_name:
                categories["file_mapped_kb"] += current_size_kb
            else:
                categories["other_kb"] += current_size_kb
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return categories


def read_mallinfo2() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        import ctypes

        class Mallinfo2(ctypes.Structure):
            _fields_ = [
                ("arena", ctypes.c_size_t),
                ("ordblks", ctypes.c_size_t),
                ("smblks", ctypes.c_size_t),
                ("hblks", ctypes.c_size_t),
                ("hblkhd", ctypes.c_size_t),
                ("usmblks", ctypes.c_size_t),
                ("fsmblks", ctypes.c_size_t),
                ("uordblks", ctypes.c_size_t),
                ("fordblks", ctypes.c_size_t),
                ("keepcost", ctypes.c_size_t),
            ]

        libc = ctypes.CDLL("libc.so.6")
        if not hasattr(libc, "mallinfo2"):
            out["unavailable_reason"] = "mallinfo2_not_exported"
            return out
        info = Mallinfo2()
        libc.mallinfo2(ctypes.byref(info))
        out.update(
            {
                "available": True,
                "arena_bytes": int(info.arena),
                "uordblks_bytes": int(info.uordblks),
                "fordblks_bytes": int(info.fordblks),
                "hblkhd_bytes": int(info.hblkhd),
                "ordblks": int(info.ordblks),
            }
        )
    except Exception as exc:
        out["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return out


def read_cuda_allocator_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cuda_available": False,
        "host_memory_stats_available": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            out["unavailable_reason"] = "cuda_not_available"
            return out
        out["cuda_available"] = True
        gpu_stats = torch.cuda.memory_stats()
        out["cuda_gpu_allocated_bytes"] = int(gpu_stats.get("allocated_bytes.all.current", 0))
        out["cuda_gpu_reserved_bytes"] = int(gpu_stats.get("reserved_bytes.all.current", 0))
        out["cuda_gpu_stats_role"] = "negative_control_not_host_rss_contributor"
        if hasattr(torch.cuda, "host_memory_stats"):
            try:
                host_stats = torch.cuda.host_memory_stats()
                out["host_memory_stats_available"] = True
                for key in (
                    "allocated_bytes.all.current",
                    "active_bytes.all.current",
                    "reserved_bytes.all.current",
                    "inactive_split_bytes.all.current",
                ):
                    if key in host_stats:
                        out[f"cuda_host_{key.replace('.', '_')}"] = int(host_stats[key])
            except Exception as exc:
                out["host_memory_stats_unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        else:
            out["host_memory_stats_unavailable_reason"] = "host_memory_stats_api_missing"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def read_vma_entries(*, exclude_ranges: Sequence[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    """Compact per-VMA smaps rows (no raw dump)."""
    exclude = list(exclude_ranges or [])
    entries: list[dict[str, Any]] = []
    try:
        lines = Path("/proc/self/smaps").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    current: dict[str, Any] | None = None
    for line in lines:
        if _is_vma_header_line(line):
            if current is not None:
                entries.append(current)
            parts = line.split()
            addr_range = parts[0]
            if "-" not in addr_range:
                continue
            start_hex, end_hex = addr_range.split("-", 1)
            try:
                start = int(start_hex, 16)
                end = int(end_hex, 16)
            except ValueError:
                current = None
                continue
            name = parts[-1] if len(parts) >= 6 else ""
            skip = False
            for lo, hi in exclude:
                if start >= lo and end <= hi:
                    skip = True
                    break
            current = {
                "start": start,
                "end": end,
                "name": name,
                "size_kb": 0,
                "rss_kb": 0,
                "private_dirty_kb": 0,
                "anonymous_kb": 0,
                "referenced_kb": 0,
                "excluded_hook_vma": skip,
            }
            continue
        if current is None or current.get("excluded_hook_vma"):
            continue
        if line.startswith("Size:"):
            current["size_kb"] = int(line.split()[1])
        elif line.startswith("Rss:"):
            current["rss_kb"] = int(line.split()[1])
        elif line.startswith("Private_Dirty:"):
            current["private_dirty_kb"] = int(line.split()[1])
        elif line.startswith("Anonymous:"):
            current["anonymous_kb"] = int(line.split()[1])
        elif line.startswith("Referenced:"):
            current["referenced_kb"] = int(line.split()[1])
    if current is not None and not current.get("excluded_hook_vma"):
        entries.append(current)
    return entries


def diff_vma_entries(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 32,
) -> dict[str, Any]:
    before_by_key = {
        (int(row["start"]), int(row["end"])): dict(row)
        for row in before
        if row.get("start") is not None and row.get("end") is not None
    }
    after_by_key = {
        (int(row["start"]), int(row["end"])): dict(row)
        for row in after
        if row.get("start") is not None and row.get("end") is not None
    }
    deltas: list[dict[str, Any]] = []
    for key, post in after_by_key.items():
        pre = before_by_key.get(key, {})
        delta_rss = int(post.get("rss_kb") or 0) - int(pre.get("rss_kb") or 0)
        delta_anon = int(post.get("anonymous_kb") or 0) - int(pre.get("anonymous_kb") or 0)
        if delta_rss <= 0 and delta_anon <= 0:
            continue
        deltas.append(
            {
                "start": key[0],
                "end": key[1],
                "name": post.get("name"),
                "delta_rss_kb": delta_rss,
                "delta_anonymous_kb": delta_anon,
                "delta_private_dirty_kb": int(post.get("private_dirty_kb") or 0)
                - int(pre.get("private_dirty_kb") or 0),
                "is_new_vma": key not in before_by_key,
            }
        )
    deltas.sort(key=lambda row: int(row.get("delta_rss_kb") or 0), reverse=True)
    return {
        "top_positive_vma_deltas": deltas[:top_k],
        "total_positive_rss_delta_kb": sum(int(row.get("delta_rss_kb") or 0) for row in deltas if row["delta_rss_kb"] > 0),
        "total_positive_anonymous_delta_kb": sum(
            int(row.get("delta_anonymous_kb") or 0) for row in deltas if row["delta_anonymous_kb"] > 0
        ),
    }


def snapshot_allocator_probe(
    *,
    exclude_hook_vmas: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    vma_entries = read_vma_entries(exclude_ranges=exclude_hook_vmas)
    return {
        "smaps_rollup": read_smaps_rollup(),
        "smaps_categories": read_smaps_categories(),
        "vma_entries": vma_entries,
        "mallinfo2": read_mallinfo2(),
        "cuda_allocator": read_cuda_allocator_stats(),
    }


def empty_cuda_host_and_device_cache() -> dict[str, Any]:
    """Confirmatory host-cache intervention (FOLD B — marked diagnostic only)."""
    pre = snapshot_allocator_probe()
    pre_rss = _proc_kib_field(Path("/proc/self/status"), "VmRSS:")
    try:
        import torch

        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status = "ok"
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
    post = snapshot_allocator_probe()
    post_rss = _proc_kib_field(Path("/proc/self/status"), "VmRSS:")
    trim_delta_kb = None
    if pre_rss is not None and post_rss is not None:
        trim_delta_kb = int(pre_rss) - int(post_rss)
    return {
        "status": status,
        "pre_rss_kib": pre_rss,
        "post_rss_kib": post_rss,
        "trim_delta_rss_kib": trim_delta_kb,
        "pre_probe": pre,
        "post_probe": post,
    }
