"""Host allocator / smaps probe for non-torch RSS attribution (compact scalars only)."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES = 4 * 1024 * 1024
ALLOCATOR_TYPE_RECONCILE_MIN = 0.8
ALLOCATOR_TYPE_RECONCILE_MAX = 1.2
ALLOCATOR_TYPE_DOMINANCE = 0.80
DISJOINT_OVERLAP_TOLERANCE_BYTES = 64 * 1024 * 1024
DISJOINTNESS_PROBE_ALLOC_BYTES = 256 * 1024 * 1024

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


def _capture_malloc_info_xml() -> tuple[bytes | None, str | None]:
    libc: ctypes.CDLL | None = None
    buf = ctypes.c_void_p()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        if not hasattr(libc, "malloc_info"):
            return None, "malloc_info_not_exported"
        libc.open_memstream.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        libc.open_memstream.restype = ctypes.c_void_p
        libc.fclose.argtypes = [ctypes.c_void_p]
        libc.free.argtypes = [ctypes.c_void_p]
        size = ctypes.c_size_t()
        fp = libc.open_memstream(ctypes.byref(buf), ctypes.byref(size))
        if not fp:
            return None, "open_memstream_failed"
        rc = int(libc.malloc_info(0, fp))
        libc.fclose(fp)
        if rc != 0:
            return None, f"malloc_info_rc_{rc}"
        return ctypes.string_at(buf.value, size.value), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if libc is not None and buf.value:
            libc.free(buf)


def parse_malloc_info_xml(raw: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        out["unavailable_reason"] = f"xml_parse_error: {exc}"
        return out
    totals: dict[str, int] = {}
    for total in root.findall("./total"):
        kind = str(total.get("type") or "")
        if not kind:
            continue
        totals[f"total_{kind}_bytes"] = int(total.get("size") or 0)
    system_current = None
    for system in root.findall("./system"):
        if str(system.get("type") or "") == "current":
            system_current = int(system.get("size") or 0)
            break
    per_heap_system: list[int] = []
    for heap in root.findall("./heap"):
        for system in heap.findall("./system"):
            if str(system.get("type") or "") == "current":
                per_heap_system.append(int(system.get("size") or 0))
    if system_current is None and not totals:
        out["unavailable_reason"] = "malloc_info_missing_components"
        return out
    out.update(
        {
            "available": True,
            "total_rest_bytes": totals.get("total_rest_bytes"),
            "total_mmap_bytes": totals.get("total_mmap_bytes"),
            "total_fast_bytes": totals.get("total_fast_bytes"),
            "system_current_bytes": system_current,
            "per_heap_system_current_bytes": per_heap_system,
            "glibc_arena_system_or_retained_bytes": system_current,
            "raw_components": totals,
            "label": "glibc_arena_system_or_retained",
        }
    )
    return out


def read_malloc_info_all_arenas() -> dict[str, Any]:
    raw, err = _capture_malloc_info_xml()
    if raw is None:
        return {"available": False, "unavailable_reason": err or "malloc_info_capture_failed"}
    parsed = parse_malloc_info_xml(raw)
    if not parsed.get("available"):
        return parsed
    parsed["xml_bytes"] = len(raw)
    return parsed


def measure_malloc_info_self_footprint(*, samples: int = 3) -> dict[str, Any]:
    readings: list[dict[str, Any]] = []
    for _ in range(max(int(samples), 2)):
        readings.append(read_malloc_info_all_arenas())
    if not all(row.get("available") for row in readings):
        return {
            "status": "unavailable",
            "malloc_info_self_footprint_status": "malloc_info_unavailable",
            "malloc_info_self_footprint_bytes": None,
            "readings": readings,
        }
    systems = [
        int(row.get("system_current_bytes") or 0)
        for row in readings
        if row.get("system_current_bytes") is not None
    ]
    mmaps = [
        int(row.get("total_mmap_bytes") or 0)
        for row in readings
        if row.get("total_mmap_bytes") is not None
    ]
    rests = [
        int(row.get("total_rest_bytes") or 0)
        for row in readings
        if row.get("total_rest_bytes") is not None
    ]
    sys_delta = max(systems) - min(systems) if systems else 0
    mmap_delta = max(mmaps) - min(mmaps) if mmaps else 0
    rest_delta = max(rests) - min(rests) if rests else 0
    footprint = max(sys_delta, mmap_delta, rest_delta)
    status = "ok" if footprint <= MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES else "exceeded"
    return {
        "status": status,
        "malloc_info_self_footprint_status": status,
        "malloc_info_self_footprint_bytes": int(footprint),
        "malloc_info_self_footprint_threshold_bytes": MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES,
        "samples": int(samples),
        "readings": readings,
    }


def _delta_bytes(exit_val: int | None, enter_val: int | None) -> int | None:
    if exit_val is None or enter_val is None:
        return None
    return int(exit_val) - int(enter_val)


def compute_delta_disjoint_partition(
    *,
    c4_delta_rss_bytes: int,
    hook_window_net_bytes: int,
    malloc_info_enter: Mapping[str, Any],
    malloc_info_exit: Mapping[str, Any],
    mmap_hook_catches_glibc_internal: bool | None,
    self_footprint_bytes: int = 0,
    cuda_host_delta_bytes: int | None = None,
    cuda_host_measured: bool = False,
) -> dict[str, Any]:
    enter_mmap = malloc_info_enter.get("total_mmap_bytes")
    exit_mmap = malloc_info_exit.get("total_mmap_bytes")
    enter_sys = malloc_info_enter.get("system_current_bytes")
    exit_sys = malloc_info_exit.get("system_current_bytes")
    delta_glibc_mmap = _delta_bytes(
        int(exit_mmap) if exit_mmap is not None else None,
        int(enter_mmap) if enter_mmap is not None else None,
    )
    delta_glibc_arena = _delta_bytes(
        int(exit_sys) if exit_sys is not None else None,
        int(enter_sys) if enter_sys is not None else None,
    )
    if delta_glibc_arena is not None:
        delta_glibc_arena = max(int(delta_glibc_arena) - int(self_footprint_bytes), 0)

    fail_reasons: list[str] = []
    if delta_glibc_mmap is None or enter_mmap is None or exit_mmap is None:
        fail_reasons.append("missing_malloc_info_mmap_component")
    if delta_glibc_arena is None or enter_sys is None or exit_sys is None:
        fail_reasons.append("missing_malloc_info_system_component")
    if mmap_hook_catches_glibc_internal is None:
        fail_reasons.append("disjointness_probe_ambiguous")

    overlap_bytes = 0
    non_glibc_mmap: int | None = None
    if delta_glibc_mmap is not None:
        if mmap_hook_catches_glibc_internal:
            if delta_glibc_mmap > hook_window_net_bytes + DISJOINT_OVERLAP_TOLERANCE_BYTES:
                fail_reasons.append("delta_glibc_mmap_exceeds_window_net")
            non_glibc_mmap = int(hook_window_net_bytes) - int(delta_glibc_mmap)
            overlap_bytes = int(delta_glibc_mmap)
            if non_glibc_mmap < -DISJOINT_OVERLAP_TOLERANCE_BYTES:
                fail_reasons.append("negative_non_glibc_mmap")
        else:
            non_glibc_mmap = int(hook_window_net_bytes)
            overlap_bytes = 0
    else:
        non_glibc_mmap = None

    measured: dict[str, int | None] = {
        "glibc_arena_system_or_retained_bytes": delta_glibc_arena,
        "non_glibc_mmap_bytes": non_glibc_mmap,
        "cuda_host_bytes": cuda_host_delta_bytes if cuda_host_measured else None,
    }
    measured_sum = sum(int(v) for v in measured.values() if v is not None)
    residual = int(c4_delta_rss_bytes) - int(measured_sum)
    residual_fraction = (
        float(residual) / float(c4_delta_rss_bytes) if c4_delta_rss_bytes > 0 else None
    )
    reconcile_ratio = (
        float(measured_sum) / float(c4_delta_rss_bytes) if c4_delta_rss_bytes > 0 else None
    )

    return {
        "c4_delta_rss_bytes": int(c4_delta_rss_bytes),
        "hook_window_net_bytes": int(hook_window_net_bytes),
        "delta_glibc_mmap_bytes": delta_glibc_mmap,
        "delta_glibc_arena_system_or_retained_bytes": delta_glibc_arena,
        "overlap_bytes": overlap_bytes,
        "mmap_hook_catches_glibc_internal": mmap_hook_catches_glibc_internal,
        "non_glibc_mmap_bytes": non_glibc_mmap,
        "cuda_host_measured": cuda_host_measured,
        "cuda_host_delta_bytes": cuda_host_delta_bytes,
        "measured_buckets_bytes": measured,
        "measured_sum_bytes": int(measured_sum),
        "residual_unattributed_bytes": int(residual),
        "residual_fraction_of_c4": residual_fraction,
        "reconcile_ratio_measured_only": reconcile_ratio,
        "fail_reasons": fail_reasons,
    }


def run_mmap_disjointness_probe_worker(
    *,
    alloc_bytes: int = DISJOINTNESS_PROBE_ALLOC_BYTES,
    stats_path: str,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        arm_hook,
        disarm_hook,
        flush_stats,
        prefault_hook,
        profile_alloc_hook_enabled,
    )

    if not profile_alloc_hook_enabled():
        return {"status": "HOOK_FAILURE", "reason": "alloc_hook_env_disabled"}
    pref = prefault_hook()
    if not pref.get("prefault_done"):
        return {"status": "HOOK_FAILURE", "reason": "prefault_failed", "prefault": pref}
    before_info = read_malloc_info_all_arenas()
    arm = arm_hook()
    if not arm.get("recording_armed"):
        return {"status": "HOOK_FAILURE", "reason": "arm_failed", "arm": arm}

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    ptr = libc.malloc(ctypes.c_size_t(alloc_bytes))
    if not ptr:
        disarm_hook()
        return {"status": "HOOK_FAILURE", "reason": "probe_malloc_failed"}
    touch_stride = 1024 * 1024
    for offset in range(0, alloc_bytes, touch_stride):
        ctypes.c_char.from_address(int(ptr) + offset).value = 1
    after_info = read_malloc_info_all_arenas()
    stats = flush_stats(Path(stats_path))
    disarm_hook()
    libc.free(ptr)

    hook_window = int(stats.get("window_net_bytes") or 0)
    delta_mmap = _delta_bytes(
        int(after_info.get("total_mmap_bytes") or 0)
        if after_info.get("total_mmap_bytes") is not None
        else None,
        int(before_info.get("total_mmap_bytes") or 0)
        if before_info.get("total_mmap_bytes") is not None
        else None,
    )
    catches = hook_window > 64 * 1024 * 1024 and (
        delta_mmap is None or hook_window >= max(int(delta_mmap), 64 * 1024 * 1024)
    )
    return {
        "status": "ok",
        "alloc_bytes": int(alloc_bytes),
        "hook_window_net_bytes": hook_window,
        "malloc_info_before": before_info,
        "malloc_info_after": after_info,
        "delta_malloc_info_mmap_bytes": delta_mmap,
        "mmap_hook_catches_glibc_internal": bool(catches),
        "stats": stats,
    }


def run_isolated_mmap_disjointness_probe(
    *,
    so_path: Path,
    out_path: Path | None = None,
    alloc_bytes: int = DISJOINTNESS_PROBE_ALLOC_BYTES,
) -> dict[str, Any]:
    if not so_path.is_file():
        return {"status": "HOOK_FAILURE", "reason": "hook_so_missing", "path": str(so_path)}
    if not read_malloc_info_all_arenas().get("available"):
        return {
            "status": "INCONCLUSIVE",
            "reason": "malloc_info_unavailable",
            "malloc_info": read_malloc_info_all_arenas(),
        }
    stats_path = (
        out_path.parent / "disjointness_probe_stats.json"
        if out_path is not None
        else Path("/tmp/disjointness_probe_stats.json")
    )
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(so_path)
    env["HRM_TEXT_158_PROFILE_ALLOC_HOOK"] = "1"
    env["HRM_TEXT_158_PROFILE_HOST_RSS"] = "1"
    env["HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"] = str(stats_path)
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; "
                "from pathlib import Path; "
                "from calm.hrm_text_158.native_full_stack.host_allocator_probe import run_mmap_disjointness_probe_worker; "
                "out=run_mmap_disjointness_probe_worker("
                "alloc_bytes=int(os.environ['PROBE_ALLOC_BYTES']), "
                f"stats_path={str(stats_path)!r}); "
                "print(json.dumps(out))"
            ),
        ],
        env={**env, "PROBE_ALLOC_BYTES": str(int(alloc_bytes))},
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        payload = {
            "status": "INCONCLUSIVE",
            "reason": "disjointness_subprocess_failed",
            "exit_code": int(proc.returncode),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        }
    payload["subprocess_exit_code"] = int(proc.returncode)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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


def capture_debugmallocstats_fd2() -> tuple[str | None, str | None]:
    """Capture sys._debugmallocstats() via fd-2 dup2 (C-level stderr)."""
    import os
    import sys
    import tempfile

    if not hasattr(sys, "_debugmallocstats"):
        return None, "debugmallocstats_not_exported"
    saved_fd = os.dup(2)
    capture_fd: int | None = None
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
            tmp_path = tmp.name
        capture_fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.dup2(capture_fd, 2)
        sys.stdout.flush()
        sys.stderr.flush()
        sys._debugmallocstats()
        sys.stdout.flush()
        sys.stderr.flush()
        text = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return None, "debugmallocstats_empty_capture"
        return text, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.dup2(saved_fd, 2)
        except Exception:
            pass
        if capture_fd is not None:
            try:
                os.close(capture_fd)
            except Exception:
                pass
        try:
            os.close(saved_fd)
        except Exception:
            pass
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


_DEBUGMALLOC_ARENA_CURRENT_RE = re.compile(r"# arenas allocated current\s*=\s*(\d+)")
_DEBUGMALLOC_ARENAS_TOTAL_RE = re.compile(r"# arenas allocated total\s*=\s*(\d+)")
_DEBUGMALLOC_ARENAS_HIGHWATER_RE = re.compile(r"# arenas highwater mark\s*=\s*(\d+)")
_DEBUGMALLOC_ARENA_CURRENT_BYTES_RE = re.compile(
    r"(\d+) arenas \* (\d+) bytes/arena\s*=\s*([\d,]+)"
)
_DEBUGMALLOC_ALLOCATED_BLOCKS_RE = re.compile(
    r"# bytes in allocated blocks\s*=\s*([\d,]+)"
)
_DEBUGMALLOC_AVAILABLE_BLOCKS_RE = re.compile(
    r"# bytes in available blocks\s*=\s*([\d,]+)"
)
DEBUGMALLOC_SELF_FOOTPRINT_MAX_BYTES = 4 * 1024 * 1024


def _parse_debugmalloc_int(value: str) -> int:
    return int(str(value).replace(",", ""))


def parse_debugmallocstats(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"raw_line_count": len(text.splitlines())}
    current_match = _DEBUGMALLOC_ARENA_CURRENT_RE.search(text)
    if current_match:
        out["arenas_allocated_current"] = int(current_match.group(1))
        out["arena_count"] = int(current_match.group(1))
    total_match = _DEBUGMALLOC_ARENAS_TOTAL_RE.search(text)
    if total_match:
        out["arenas_allocated_total"] = int(total_match.group(1))
    highwater_match = _DEBUGMALLOC_ARENAS_HIGHWATER_RE.search(text)
    if highwater_match:
        out["arenas_highwater_mark"] = int(highwater_match.group(1))
    arena_bytes_match = _DEBUGMALLOC_ARENA_CURRENT_BYTES_RE.search(text)
    if arena_bytes_match:
        out["arenas_in_product_line"] = int(arena_bytes_match.group(1))
        out["bytes_per_arena"] = int(arena_bytes_match.group(2))
        out["arena_bytes"] = _parse_debugmalloc_int(arena_bytes_match.group(3))
    allocated_blocks_match = _DEBUGMALLOC_ALLOCATED_BLOCKS_RE.search(text)
    if allocated_blocks_match:
        out["bytes_in_allocated_blocks"] = _parse_debugmalloc_int(
            allocated_blocks_match.group(1)
        )
    available_blocks_match = _DEBUGMALLOC_AVAILABLE_BLOCKS_RE.search(text)
    if available_blocks_match:
        out["bytes_in_available_blocks"] = _parse_debugmalloc_int(
            available_blocks_match.group(1)
        )
    if (
        out.get("arenas_allocated_current") is not None
        and out.get("arenas_in_product_line") is not None
        and int(out["arenas_allocated_current"]) != int(out["arenas_in_product_line"])
    ):
        out["arena_bytes_current_mismatch"] = True
    required_present = (
        out.get("arena_bytes") is not None
        and out.get("bytes_in_allocated_blocks") is not None
        and out.get("arenas_allocated_current") is not None
    )
    if required_present:
        out["parse_ok"] = True
    else:
        out["parse_ok"] = False
        out["unavailable_reason"] = "debugmallocstats_missing_required_fields"
    return out


def read_debugmallocstats() -> dict[str, Any]:
    text, err = capture_debugmallocstats_fd2()
    if err or not text:
        return {
            "available": False,
            "unavailable_reason": err or "debugmallocstats_capture_failed",
        }
    parsed = parse_debugmallocstats(text)
    if not parsed.get("parse_ok"):
        return {
            "available": False,
            "unavailable_reason": parsed.get("unavailable_reason"),
            "raw_preview": text[:200],
        }
    return {"available": True, **parsed}


def preflight_debugmallocstats_self_test() -> dict[str, Any]:
    result = read_debugmallocstats()
    return {
        "status": "ok" if result.get("available") else "unavailable",
        "capture_method": "os.dup2_fd2_tempfile",
        "debugmallocstats": result,
    }


def profile_debugmallocstats_enabled() -> bool:
    import os

    rss_on = os.environ.get("HRM_TEXT_158_PROFILE_HOST_RSS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dm_on = os.environ.get("HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return rss_on and dm_on


def measure_debugmallocstats_self_footprint(*, samples: int = 3) -> dict[str, Any]:
    readings: list[dict[str, Any]] = []
    for _ in range(max(int(samples), 2)):
        readings.append(read_debugmallocstats())
    if not all(row.get("available") for row in readings):
        return {
            "status": "unavailable",
            "debugmallocstats_self_footprint_status": "debugmallocstats_unavailable",
            "debugmallocstats_self_footprint_bytes": None,
            "readings": readings,
        }
    arena_bytes = [
        int(row.get("arena_bytes") or 0)
        for row in readings
        if row.get("arena_bytes") is not None
    ]
    allocated_blocks = [
        int(row.get("bytes_in_allocated_blocks") or 0)
        for row in readings
        if row.get("bytes_in_allocated_blocks") is not None
    ]
    arena_delta = max(arena_bytes) - min(arena_bytes) if arena_bytes else 0
    allocated_delta = (
        max(allocated_blocks) - min(allocated_blocks) if allocated_blocks else 0
    )
    footprint = max(arena_delta, allocated_delta)
    status = (
        "ok" if footprint <= DEBUGMALLOC_SELF_FOOTPRINT_MAX_BYTES else "exceeded"
    )
    return {
        "status": status,
        "debugmallocstats_self_footprint_status": status,
        "debugmallocstats_self_footprint_bytes": int(footprint),
        "debugmallocstats_self_footprint_threshold_bytes": DEBUGMALLOC_SELF_FOOTPRINT_MAX_BYTES,
        "samples": int(samples),
        "readings": readings,
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
        "malloc_info_all_arenas": read_malloc_info_all_arenas(),
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
