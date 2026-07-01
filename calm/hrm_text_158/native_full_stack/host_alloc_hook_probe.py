"""Alloc-hook interposer probe facade (compact scalars only)."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

PROFILE_ALLOC_HOOK_ENV = "HRM_TEXT_158_PROFILE_ALLOC_HOOK"
ALLOC_HOOK_LOG_ENV = "HRM_TEXT_158_ALLOC_HOOK_LOG_PATH"
ALLOC_HOOK_STATS_ENV = "HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"

SKIP_MODULE_FRAGMENTS = (
    "libhrm_alloc_hook",
    "/libc.",
    "ld-linux",
    "libdl.",
    "libpthread",
    "libgcc",
    "libstdc++",
)

HOOK_RESOLVE_DOMINANCE = 0.80
HOOK_RECONCILE_MIN = 0.8
HOOK_RECONCILE_MAX = 1.2
UNKNOWN_FREE_MAX_FRACTION = 0.10
HOOK_INSTRUMENTATION_MAX_GIB = 0.25


def profile_alloc_hook_enabled() -> bool:
    rss_on = os.environ.get("HRM_TEXT_158_PROFILE_HOST_RSS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    hook_on = os.environ.get(PROFILE_ALLOC_HOOK_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return rss_on and hook_on


def default_hook_so_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "alloc_hook"
        / "libhrm_alloc_hook.so"
    )


def load_hook_library(so_path: Path | None = None) -> ctypes.CDLL | None:
    path = so_path or default_hook_so_path()
    preload = os.environ.get("LD_PRELOAD", "")
    if str(path) in preload or path.name in preload:
        try:
            return ctypes.CDLL(None)
        except OSError:
            pass
    if not path.is_file():
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        return None


def _bind_hook_api(lib: ctypes.CDLL) -> None:
    lib.hrm_alloc_hook_is_active.restype = ctypes.c_int
    lib.hrm_alloc_hook_is_recording.restype = ctypes.c_int
    lib.hrm_alloc_hook_arm.restype = None
    lib.hrm_alloc_hook_disarm.restype = None
    lib.hrm_alloc_hook_prefault.restype = ctypes.c_int
    lib.hrm_alloc_hook_reset_aggregation_window.restype = None
    lib.hrm_alloc_hook_flush_stats_json.argtypes = [ctypes.c_char_p]
    lib.hrm_alloc_hook_flush_stats_json.restype = ctypes.c_int
    flush_live = getattr(lib, "hrm_alloc_hook_flush_live_ranges_json", None)
    if flush_live is not None:
        flush_live.argtypes = [ctypes.c_char_p]
        flush_live.restype = ctypes.c_int
    lib.hrm_alloc_hook_note_positive_control.argtypes = [ctypes.c_uint64]
    lib.hrm_alloc_hook_note_positive_control.restype = None


def hook_vma_ranges(stats: Mapping[str, Any]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for start_key, end_key in (
        ("hook_table_start", "hook_table_end"),
        ("hook_ring_start", "hook_ring_end"),
    ):
        start = stats.get(start_key)
        end = stats.get(end_key)
        if start is None or end is None:
            continue
        ranges.append((int(start), int(end)))
    return ranges


def read_stats_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"error": "stats_missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def flush_live_ranges(path: Path, *, so_path: Path | None = None) -> dict[str, Any]:
    lib = load_hook_library(so_path)
    if lib is None:
        return {"status": "hook_so_missing"}
    _bind_hook_api(lib)
    flush_live = getattr(lib, "hrm_alloc_hook_flush_live_ranges_json", None)
    if flush_live is None:
        return {"status": "live_ranges_api_missing"}
    path.parent.mkdir(parents=True, exist_ok=True)
    rc = int(flush_live(str(path).encode("utf-8")))
    if rc != 0:
        return {"status": "flush_failed", "rc": rc}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "parse_failed", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "live_ranges": list(payload.get("live_ranges") or [])}


def flush_stats(path: Path, *, so_path: Path | None = None) -> dict[str, Any]:
    lib = load_hook_library(so_path)
    if lib is None:
        return {"status": "hook_so_missing"}
    _bind_hook_api(lib)
    if int(lib.hrm_alloc_hook_is_active()) != 1:
        return {"status": "hook_inactive"}
    path.parent.mkdir(parents=True, exist_ok=True)
    rc = int(lib.hrm_alloc_hook_flush_stats_json(str(path).encode("utf-8")))
    if rc != 0:
        return {"status": "flush_failed", "rc": rc}
    return read_stats_json(path)


def prefault_hook(*, so_path: Path | None = None) -> dict[str, Any]:
    lib = load_hook_library(so_path)
    if lib is None:
        return {"status": "hook_so_missing", "prefault_done": False}
    _bind_hook_api(lib)
    if not profile_alloc_hook_enabled():
        ensure = getattr(lib, "hrm_alloc_hook_prefault", None)
        if ensure is None:
            return {"status": "prefault_api_missing", "prefault_done": False}
    rc = int(lib.hrm_alloc_hook_prefault())
    return {"status": "ok" if rc == 1 else "prefault_failed", "prefault_done": rc == 1}


def reset_aggregation_window(*, so_path: Path | None = None) -> None:
    lib = load_hook_library(so_path)
    if lib is None:
        return
    _bind_hook_api(lib)
    if int(lib.hrm_alloc_hook_is_active()) == 1:
        lib.hrm_alloc_hook_reset_aggregation_window()


def arm_hook(*, so_path: Path | None = None) -> dict[str, Any]:
    lib = load_hook_library(so_path)
    if lib is None:
        return {"status": "hook_so_missing", "recording_armed": False}
    _bind_hook_api(lib)
    lib.hrm_alloc_hook_arm()
    armed = int(lib.hrm_alloc_hook_is_recording()) == 1
    return {"status": "ok" if armed else "arm_failed", "recording_armed": armed}


def disarm_hook(*, so_path: Path | None = None) -> dict[str, Any]:
    lib = load_hook_library(so_path)
    if lib is None:
        return {"status": "hook_so_missing", "recording_armed": False}
    _bind_hook_api(lib)
    lib.hrm_alloc_hook_disarm()
    armed = int(lib.hrm_alloc_hook_is_recording()) == 1
    return {"status": "ok", "recording_armed": armed}


def run_ld_preload_torch_preflight(so_path: Path | None = None) -> dict[str, Any]:
    path = so_path or default_hook_so_path()
    if not path.is_file():
        return {"status": "HOOK_FAILURE", "reason": "hook_so_missing", "path": str(path)}
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(path)
    env[PROFILE_ALLOC_HOOK_ENV] = "1"
    env["HRM_TEXT_158_PROFILE_HOST_RSS"] = "1"
    proc = subprocess.run(
        [
            "python3",
            "-c",
            "import torch; print('torch_ok', torch.__version__)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        return {
            "status": "HOOK_FAILURE",
            "reason": "ld_preload_torch_import_failed",
            "exit_code": int(proc.returncode),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        }
    return {"status": "ok", "stdout": proc.stdout.strip()}


def run_positive_control(
    stats_path: Path,
    *,
    malloc_bytes: int = 1_048_576,
    torch_bytes: int = 2_097_152,
    aligned_bytes: int = 4_194_304,
) -> dict[str, Any]:
    lib = load_hook_library()
    if lib is None:
        return {"status": "HOOK_FAILURE", "reason": "hook_so_missing"}
    _bind_hook_api(lib)
    if int(lib.hrm_alloc_hook_is_active()) != 1:
        pref = prefault_hook(so_path=default_hook_so_path())
        if not pref.get("prefault_done"):
            return {"status": "HOOK_FAILURE", "reason": "hook_prefault_failed", "prefault": pref}
    if int(lib.hrm_alloc_hook_is_active()) != 1:
        return {"status": "HOOK_FAILURE", "reason": "hook_inactive"}

    arm_result = arm_hook()
    if not arm_result.get("recording_armed"):
        return {"status": "HOOK_FAILURE", "reason": "hook_arm_failed", "arm": arm_result}

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    checks: list[dict[str, Any]] = []

    ptr = libc.malloc(ctypes.c_size_t(malloc_bytes))
    if not ptr:
        return {"status": "HOOK_FAILURE", "reason": "malloc_failed"}
    lib.hrm_alloc_hook_note_positive_control(ctypes.c_uint64(malloc_bytes))
    checks.append({"kind": "malloc", "bytes": malloc_bytes, "ptr": hex(int(ptr))})
    libc.free(ptr)

    import torch

    tensor = torch.empty(torch_bytes // 4, dtype=torch.float32)
    checks.append({"kind": "torch_cpu_tensor", "bytes": int(tensor.numel() * tensor.element_size())})
    del tensor

    aligned_ptr = ctypes.c_void_p()
    rc = libc.posix_memalign(
        ctypes.byref(aligned_ptr),
        ctypes.c_size_t(64),
        ctypes.c_size_t(aligned_bytes),
    )
    if rc != 0:
        return {"status": "HOOK_FAILURE", "reason": "posix_memalign_failed", "rc": int(rc)}
    checks.append({"kind": "posix_memalign", "bytes": aligned_bytes, "ptr": hex(int(aligned_ptr.value or 0))})
    libc.free(aligned_ptr)

    stats = flush_stats(stats_path)
    disarm_hook()
    if stats.get("error"):
        return {"status": "HOOK_FAILURE", "reason": "stats_flush_failed", "stats": stats}
    return {
        "status": "ok",
        "kind": "hook_self_control_cpu",
        "checks": checks,
        "stats": stats,
        "positive_control_passed": True,
    }


def symbolize_frame(owner_frame: str | int, *, so_path: Path | None = None) -> dict[str, Any]:
    frame = str(owner_frame)
    if frame.startswith("0x"):
        addr = frame
    else:
        addr = hex(int(owner_frame))
    path = so_path or default_hook_so_path()
    proc = subprocess.run(
        ["addr2line", "-f", "-C", "-e", str(path), addr],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or proc.stdout.strip() == "??:0":
        return {"owner_frame": frame, "resolved": False, "symbol": None, "module": str(path.name)}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    symbol = lines[0] if lines else None
    location = lines[1] if len(lines) > 1 else None
    return {
        "owner_frame": frame,
        "resolved": bool(location and location != "??:0"),
        "symbol": symbol,
        "location": location,
        "module": str(path.name),
    }


def attribute_alloc_hook_profile(
    hook_marks: Sequence[Mapping[str, Any]],
    *,
    c4_delta_rss_gib: float | None,
) -> dict[str, Any]:
    enter = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_C4_enter"), None)
    exit_mark = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_C4_exit"), None)
    preflight = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_preflight"), None)

    if preflight and str(preflight.get("status")) == "HOOK_FAILURE":
        return {
            "status": "HOOK_FAILURE",
            "mechanism_owner_status": "HOOK_FAILURE",
            "tier": "HOOK_FAILURE",
            "call_site_status": "UNRESOLVED",
            "preflight": dict(preflight),
            "hook_ran": False,
        }

    stats_enter = dict((enter or {}).get("alloc_hook_stats") or {})
    stats_exit = dict((exit_mark or {}).get("alloc_hook_stats") or {})
    top_sites = list(stats_exit.get("top_sites") or [])

    dominant = top_sites[0] if top_sites else None
    dominant_net = float(dominant.get("net_bytes") or 0) if dominant else 0.0
    window_net = float(stats_exit.get("window_net_bytes") or 0)
    lost_owner = int(stats_exit.get("lost_owner_count") or 0)
    ring_drops = int(stats_exit.get("ring_drop_count") or 0)
    lock_contention_drops = int(stats_exit.get("lock_contention_drop_count") or 0)
    table_overflow = int(stats_exit.get("table_overflow_count") or 0)
    unknown_free_bytes = int(stats_exit.get("unknown_free_bytes") or 0)
    unknown_unmeasured = int(stats_exit.get("unknown_free_unmeasured_count") or 0)
    unknown_bounded = bool(stats_exit.get("unknown_free_bytes_bounded", False))

    hook_instr_delta = None
    c4_window_rss_delta_gib = None
    if enter and exit_mark:
        pre_rss = int(((enter.get("resource_snapshot") or {}).get("rss_kib")) or 0)
        post_rss = int(((exit_mark.get("resource_snapshot") or {}).get("rss_kib")) or 0)
        c4_window_rss_delta_gib = (post_rss - pre_rss) / (1024.0 * 1024.0)
        if bool(stats_exit.get("prefault_done")):
            hook_instr_delta = 0.0
        else:
            hook_instr_delta = c4_window_rss_delta_gib

    mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
    tier = "C"
    status = "UNMAPPED_ANONYMOUS_MMAP"
    allocation_source: str | None = None
    call_site_status = "UNRESOLVED"
    call_site_origin: str | None = None
    hook_ran = bool(enter and exit_mark)

    if lost_owner > 0 or table_overflow > 0 or lock_contention_drops > 0:
        mechanism_owner_status = "INCONCLUSIVE"
        tier = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
    elif not unknown_bounded or unknown_unmeasured > 0:
        mechanism_owner_status = "INCONCLUSIVE"
        tier = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
    elif c4_delta_rss_gib is not None and unknown_free_bytes > UNKNOWN_FREE_MAX_FRACTION * c4_delta_rss_gib * (1024 ** 3):
        mechanism_owner_status = "INCONCLUSIVE"
        tier = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
    elif hook_instr_delta is not None and hook_instr_delta > HOOK_INSTRUMENTATION_MAX_GIB:
        mechanism_owner_status = "INCONCLUSIVE"
        tier = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
    elif dominant and c4_delta_rss_gib and c4_delta_rss_gib > 0:
        reconcile = abs(dominant_net) / (c4_delta_rss_gib * (1024 ** 3))
        dominance = abs(dominant_net) / max(abs(window_net), 1.0)
        if (
            dominance >= HOOK_RESOLVE_DOMINANCE
            and HOOK_RECONCILE_MIN <= reconcile <= HOOK_RECONCILE_MAX
            and lost_owner == 0
            and table_overflow == 0
            and unknown_bounded
            and unknown_unmeasured == 0
        ):
            sym = symbolize_frame(str(dominant.get("owner_frame")))
            if sym.get("resolved"):
                mechanism_owner_status = "RESOLVED"
                tier = "A"
                status = "RESOLVED"
                allocation_source = str(sym.get("location"))
                call_site_status = "RESOLVED"
                call_site_origin = allocation_source
            else:
                mechanism_owner_status = "RESOLVED"
                tier = "B"
                status = "RESOLVED_MODULE_ONLY"
                allocation_source = str(sym.get("module"))
                call_site_status = "UNRESOLVED"
        elif hook_ran and window_net > 0:
            status = "UNMAPPED_ANONYMOUS_MMAP"

    window_net_gib = window_net / (1024.0 ** 3)
    c4_delta = float(c4_delta_rss_gib or 0.0)
    non_mmap_remainder_gib = max(c4_delta - window_net_gib, 0.0) if c4_delta > 0 else None
    allocator_type_partition: dict[str, Any] | None = None
    if hook_ran and c4_delta > 0:
        allocator_type_partition = {
            "c4_subphase_delta_rss_gib": c4_delta,
            "mmap_net_gib": window_net_gib,
            "mmap_net_bytes": int(window_net),
            "non_mmap_remainder_gib": non_mmap_remainder_gib,
            "non_mmap_remainder_fraction_of_c4": (
                non_mmap_remainder_gib / c4_delta if c4_delta > 0 else None
            ),
            "non_mmap_remainder_class": (
                "malloc_glibc_arena_or_driver_pinned"
                if non_mmap_remainder_gib and non_mmap_remainder_gib > 0.05
                else None
            ),
            "libc_malloc_interposition_cuda_safe": False,
            "recording_mode": "mmap_only_cuda_safe",
            "forward_fidelity_skipped_baseline_note": (
                "fixture_alloc_hook skips forward_fidelity; C4 subphase delta may differ "
                "from prior unskipped runs (e.g. 7.33 vs 7.57 GiB)"
            ),
        }

    classified_null: dict[str, Any] | None = None
    if hook_ran:
        classified_null = {
            "slice_outcome": "CLASSIFIED_NULL",
            "libc_malloc_ldpreload_cuda_incompatible": True,
            "libc_malloc_ldpreload_cuda_incompatible_status": "CONFIRMED",
            "superseded_folds": ["F4", "F7", "F11"],
            "superseded_reason": (
                "malloc-family LD_PRELOAD interception is CUDA-incompatible on this stack; "
                "pointer-owned libc ledger and unknown-free measurement folds are void"
            ),
            "surviving_mechanism": "mmap_only_recording",
            "mechanism_owner_named": mechanism_owner_status == "RESOLVED",
            "next_method_proposal": (
                "mallinfo2/malloc_stats glibc arena introspection (CUDA-safe, no interposition) "
                "+ cuda-driver probe (cuMemHostAlloc) for driver-pinned remainder"
            ),
        }

    if mechanism_owner_status == "UNMAPPED_OR_UNRESOLVED":
        call_site_status = "UNRESOLVED"
        call_site_origin = None

    return {
        "status": status,
        "mechanism_owner_status": mechanism_owner_status,
        "tier": tier,
        "allocation_source": allocation_source,
        "call_site_status": call_site_status,
        "call_site_origin_file_line": call_site_origin,
        "dominant_hook_site": dominant,
        "window_net_bytes": window_net,
        "window_net_gib": window_net_gib,
        "reconcile_ratio_vs_c4_rss": (
            abs(dominant_net) / (c4_delta_rss_gib * (1024 ** 3))
            if dominant and c4_delta_rss_gib
            else None
        ),
        "ring_drop_count": ring_drops,
        "lock_contention_drop_count": lock_contention_drops,
        "table_overflow_count": table_overflow,
        "lost_owner_count": lost_owner,
        "unknown_free_bytes": unknown_free_bytes,
        "unknown_free_unmeasured_count": unknown_unmeasured,
        "unknown_free_bytes_bounded": unknown_bounded,
        "unknown_free_fraction_of_c4_rss": (
            unknown_free_bytes / (c4_delta_rss_gib * (1024 ** 3))
            if c4_delta_rss_gib
            else None
        ),
        "hook_instrumentation_rss_delta_gib": hook_instr_delta,
        "c4_window_rss_delta_gib": c4_window_rss_delta_gib,
        "allocator_type_partition": allocator_type_partition,
        "classified_null": classified_null,
        "stats_enter": stats_enter,
        "stats_exit": stats_exit,
        "top_sites": top_sites,
        "hook_ran": hook_ran,
    }


NON_GLIBC_UNCLASSIFIED_MAX_FRACTION = 0.10
SOURCE_RESOLVE_DOMINANCE = 0.80
SOURCE_RECONCILE_MIN = 0.8
SOURCE_RECONCILE_MAX = 1.2


def _parse_addr(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    try:
        return int(text, 16) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def _is_anonymous_vma_name(name: str) -> bool:
    lowered = (name or "").lower()
    return lowered == "" or "[anon" in lowered


def _is_real_file_path_vma_name(name: str) -> bool:
    text = (name or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "[heap]" in lowered or "[stack]" in lowered or "[anon" in lowered:
        return False
    return True


def classify_vma_name(name: str, *, fd: int | None = None) -> str:
    """Classify a VMA using smaps name as authority; fd is corroboration only."""
    lowered = (name or "").lower()
    if "nvidia" in lowered or "nvidia-uvm" in lowered:
        return "cuda_driver"
    if "[heap]" in lowered:
        return "heap"
    if "[stack]" in lowered:
        return "stack"
    if _is_anonymous_vma_name(name):
        if fd is not None and fd >= 0:
            return "unknown_fd"
        return "anonymous_private"
    if _is_real_file_path_vma_name(name):
        return "file_backed"
    if fd is not None and fd == -1:
        return "anonymous_private"
    if fd is not None and fd >= 0:
        return "unknown_fd"
    return "unknown_fd"


def _interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    lo = max(start_a, start_b)
    hi = min(end_a, end_b)
    return max(0, hi - lo)


def read_proc_maps_modules() -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    try:
        for line in Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2 or "-" not in parts[0]:
                continue
            start_hex, end_hex = parts[0].split("-", 1)
            try:
                start = int(start_hex, 16)
                end = int(end_hex, 16)
            except ValueError:
                continue
            name = parts[-1] if len(parts) >= 6 else ""
            modules.append({"start": start, "end": end, "name": name, "perms": parts[1]})
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    return modules


def symbolize_frame_extended(owner_frame: str | int) -> dict[str, Any]:
    frame_int = _parse_addr(owner_frame)
    if frame_int is None:
        return {"owner_frame": str(owner_frame), "resolved": False, "tier": "C"}
    modules = read_proc_maps_modules()
    module_path: str | None = None
    module_offset: int | None = None
    for row in modules:
        if row.get("error"):
            continue
        start = int(row["start"])
        end = int(row["end"])
        if start <= frame_int < end:
            module_path = str(row.get("name") or "")
            module_offset = frame_int - start
            break
    if not module_path:
        return {
            "owner_frame": hex(frame_int),
            "resolved": False,
            "tier": "C",
            "reason": "no_maps_module",
        }
    if any(frag in module_path for frag in SKIP_MODULE_FRAGMENTS):
        return {
            "owner_frame": hex(frame_int),
            "resolved": False,
            "tier": "C",
            "module": module_path,
            "module_offset": module_offset,
            "reason": "skip_list_module",
        }
    proc = subprocess.run(
        ["addr2line", "-f", "-C", "-e", module_path, hex(module_offset or 0)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    symbol = lines[0] if lines else None
    location = lines[1] if len(lines) > 1 else None
    line_resolved = bool(location and location != "??:0")
    return {
        "owner_frame": hex(frame_int),
        "resolved": line_resolved,
        "tier": "A" if line_resolved else "B",
        "symbol": symbol,
        "location": location,
        "module": module_path,
        "module_offset": module_offset,
    }


def join_tracked_ranges_to_vmas(
    live_ranges: Sequence[Mapping[str, Any]],
    vma_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    unmapped_bytes = 0
    join_rows: list[dict[str, Any]] = []
    for rng in live_ranges:
        start = _parse_addr(rng.get("addr"))
        end = _parse_addr(rng.get("addr_end"))
        length = int(rng.get("len") or 0)
        if start is None or end is None or length <= 0:
            continue
        owner_frame = rng.get("owner_frame")
        fd = int(rng.get("fd") if rng.get("fd") is not None else -1)
        matched = False
        matched_rss = 0
        matched_name = ""
        for vma in vma_entries:
            if vma.get("excluded_hook_vma"):
                continue
            vma_start = int(vma.get("start") or 0)
            vma_end = int(vma.get("end") or 0)
            overlap = _interval_overlap(start, end, vma_start, vma_end)
            if overlap <= 0:
                continue
            matched = True
            vma_rss_kb = int(vma.get("rss_kb") or 0)
            vma_size_kb = max(int(vma.get("size_kb") or 1), 1)
            rss_share = int(vma_rss_kb * 1024 * overlap / max(vma_end - vma_start, 1))
            matched_rss += rss_share
            matched_name = str(vma.get("name") or matched_name)
            fd_class = classify_vma_name(matched_name, fd=fd)
            bucket = buckets.setdefault(
                fd_class,
                {"va_bytes": 0, "rss_bytes": 0, "owner_frames": {}},
            )
            bucket["va_bytes"] += overlap
            bucket["rss_bytes"] += rss_share
            frame_key = str(owner_frame)
            bucket["owner_frames"][frame_key] = bucket["owner_frames"].get(frame_key, 0) + overlap
        if not matched:
            unmapped_bytes += length
        join_rows.append(
            {
                "addr": hex(start),
                "addr_end": hex(end),
                "len": length,
                "fd": fd,
                "owner_frame": owner_frame,
                "vma_name": matched_name or None,
                "matched": matched,
                "matched_rss_bytes": matched_rss,
            }
        )
    owner_totals: dict[str, int] = {}
    for bucket in buckets.values():
        for frame, nbytes in bucket.get("owner_frames", {}).items():
            owner_totals[frame] = owner_totals.get(frame, 0) + int(nbytes)
    dominant_owner = None
    dominant_owner_bytes = 0
    for frame, nbytes in owner_totals.items():
        if nbytes > dominant_owner_bytes:
            dominant_owner = frame
            dominant_owner_bytes = nbytes
    return {
        "buckets": buckets,
        "join_rows": join_rows,
        "unmapped_live_range_bytes": unmapped_bytes,
        "dominant_owner_frame": dominant_owner,
        "dominant_owner_frame_bytes": dominant_owner_bytes,
        "owner_frame_totals_va_bytes": owner_totals,
    }


def attribute_non_glibc_mmap_source(
    hook_marks: Sequence[Mapping[str, Any]],
    *,
    non_glibc_mmap_target_bytes: int | None,
    allocator_type_attribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    enter = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_C4_enter"), None)
    exit_mark = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_C4_exit"), None)
    preflight = next((row for row in hook_marks if str(row.get("event")) == "alloc_hook_preflight"), None)
    if preflight and str(preflight.get("status")) == "HOOK_FAILURE":
        return {
            "status": "HOOK_FAILURE",
            "source_tier": "HOOK_FAILURE",
            "call_site_status": "UNRESOLVED",
            "hook_ran": False,
        }
    if not exit_mark:
        return {
            "status": "INCONCLUSIVE",
            "source_tier": "C",
            "call_site_status": "UNRESOLVED",
            "reason": "missing_alloc_hook_C4_exit",
            "hook_ran": False,
        }

    stats_exit = dict(exit_mark.get("alloc_hook_stats") or {})
    live_ranges = list(exit_mark.get("live_ranges") or [])
    if not live_ranges:
        live_path = exit_mark.get("live_ranges_path")
        if live_path:
            live_ranges = list(read_stats_json(Path(str(live_path))).get("live_ranges") or [])

    target_bytes = non_glibc_mmap_target_bytes
    if target_bytes is None and allocator_type_attribution:
        partition = dict(allocator_type_attribution.get("partition") or {})
        target_bytes = partition.get("non_glibc_mmap_bytes")
    if target_bytes is None:
        target_bytes = int(stats_exit.get("window_net_bytes") or 0)

    enter_probe = dict((enter or {}).get("allocator_probe") or {})
    exit_probe = dict(exit_mark.get("allocator_probe") or {})
    enter_vmas = list(enter_probe.get("vma_entries") or [])
    exit_vmas = list(exit_probe.get("vma_entries") or [])

    from calm.hrm_text_158.native_full_stack.host_allocator_probe import diff_vma_entries

    vma_diff = diff_vma_entries(enter_vmas, exit_vmas)
    join = join_tracked_ranges_to_vmas(live_ranges, exit_vmas)

    classified_va = 0
    classified_rss = 0
    fd_class_rows: list[dict[str, Any]] = []
    for fd_class, bucket in join.get("buckets", {}).items():
        va_b = int(bucket.get("va_bytes") or 0)
        rss_b = int(bucket.get("rss_bytes") or 0)
        classified_va += va_b
        classified_rss += rss_b
        fd_class_rows.append(
            {
                "fd_class": fd_class,
                "fd_class_va_bytes": va_b,
                "fd_class_rss_bytes": rss_b,
            }
        )

    unclassified = int(join.get("unmapped_live_range_bytes") or 0)
    unknown_fd_bucket = dict(join.get("buckets", {}).get("unknown_fd") or {})
    unknown_fd_bytes = int(unknown_fd_bucket.get("va_bytes") or 0)
    if unclassified == 0 and target_bytes > 0:
        unclassified = max(int(target_bytes) - classified_va, 0)

    dominant_frame = join.get("dominant_owner_frame")
    dominant_frame_va = int(join.get("dominant_owner_frame_bytes") or 0)
    sym = symbolize_frame_extended(dominant_frame) if dominant_frame else {"resolved": False, "tier": "C"}

    ring_drops = int(stats_exit.get("ring_drop_count") or 0)
    table_overflow = int(stats_exit.get("table_overflow_count") or 0)
    lost_owner = int(stats_exit.get("lost_owner_count") or 0)
    partial_munmap = int(stats_exit.get("partial_munmap_ambiguity_count") or 0)
    lock_drops = int(stats_exit.get("lock_contention_drop_count") or 0)

    fail_reasons: list[str] = []
    if unknown_fd_bytes > 0:
        fail_reasons.append("unknown_fd_classification")
    if ring_drops or table_overflow or lost_owner or lock_drops:
        fail_reasons.append("hook_drops_or_overflow")
    if partial_munmap:
        fail_reasons.append("partial_munmap_ambiguity")
    if unclassified and target_bytes > 0 and (unclassified / target_bytes) > NON_GLIBC_UNCLASSIFIED_MAX_FRACTION:
        fail_reasons.append("unclassified_over_tolerance")
    if not live_ranges:
        fail_reasons.append("no_live_ranges_captured")

    classified_live_net_bytes = classified_va
    reconcile_ratio = None
    if target_bytes > 0:
        reconcile_ratio = classified_live_net_bytes / float(target_bytes)

    source_tier = "C"
    source_status = "INCONCLUSIVE"
    call_site_status = "UNRESOLVED"
    resolved_source: str | None = None

    if fail_reasons:
        source_status = "INCONCLUSIVE"
    elif target_bytes > 0 and dominant_frame_va > 0:
        dominance = dominant_frame_va / float(max(classified_live_net_bytes, 1))
        if (
            reconcile_ratio is not None
            and SOURCE_RECONCILE_MIN <= reconcile_ratio <= SOURCE_RECONCILE_MAX
            and dominance >= SOURCE_RESOLVE_DOMINANCE
        ):
            if sym.get("tier") == "A":
                source_tier = "A"
                source_status = "RESOLVED_SOURCE"
                resolved_source = str(sym.get("location"))
                call_site_status = "RESOLVED"
            elif sym.get("tier") == "B":
                source_tier = "B"
                source_status = "RESOLVED_MODULE"
                resolved_source = str(sym.get("module"))
                call_site_status = "UNRESOLVED"
            else:
                source_status = "UNMAPPED_ANONYMOUS_MMAP"
        else:
            source_status = "INCONCLUSIVE"
            if dominant_frame and fd_class_rows and fd_class_rows[0]["fd_class"] == "anonymous_private":
                source_status = "ANONYMOUS_BUCKET_TRIVIAL_FD_CLASS"

    return {
        "status": source_status,
        "source_tier": source_tier,
        "resolved_source": resolved_source,
        "call_site_status": call_site_status,
        "non_glibc_mmap_target_bytes": int(target_bytes),
        "classified_live_net_bytes": classified_live_net_bytes,
        "classified_live_net_gib": classified_live_net_bytes / (1024.0**3),
        "unclassified_net_bytes": unclassified,
        "unclassified_net_fraction": (unclassified / target_bytes) if target_bytes else None,
        "reconcile_ratio_vs_non_glibc_target": reconcile_ratio,
        "dominant_owner_frame": dominant_frame,
        "dominant_owner_frame_symbol": sym,
        "fd_class_rows": fd_class_rows,
        "vma_rss_reconcile": vma_diff,
        "range_vma_join": join,
        "live_range_count": len(live_ranges),
        "fail_reasons": fail_reasons,
        "hook_stats_exit": stats_exit,
        "anonymous_bucket_fd_class_trivial": True,
        "recording_mode": "mmap_all_fd_cuda_safe",
        "stale_fd_readlink_authority": False,
        "classification_authority": "tracked_range_vma_join",
    }
