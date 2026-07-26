#!/usr/bin/env python3
"""Shared O_EXCL robust copy / write helpers for P1b artifact discipline."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Set, Union

PathLike = Union[str, Path]


class ShortWriteError(OSError):
    """Raised when an O_EXCL write loop cannot drain the full buffer."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_bytes_o_excl(path: PathLike, data: bytes) -> str:
    """O_EXCL-create *path*, write all of *data* with flush+fsync; return sha256."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(dest), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "wb") as f:
            fd = -1  # ownership transferred
            view = memoryview(data)
            offset = 0
            total = len(data)
            while offset < total:
                n = f.write(view[offset:])
                if not n:
                    raise ShortWriteError(
                        f"short-write at offset={offset} remaining={total - offset} path={dest}"
                    )
                offset += int(n)
            if offset != total:
                raise ShortWriteError(
                    f"incomplete write wrote={offset} expected={total} path={dest}"
                )
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    digest = _sha256_file(dest)
    if digest != _sha256_bytes(data):
        raise RuntimeError(f"post-write sha mismatch for {dest}")
    return digest


def copy_file_o_excl(src: PathLike, dst: PathLike) -> tuple[str, str]:
    """Robust O_EXCL copy with write-loop + dual sha identity check."""
    src_path = Path(src)
    data = src_path.read_bytes()
    src_sha = _sha256_bytes(data)
    dst_sha = write_bytes_o_excl(dst, data)
    if src_sha != dst_sha:
        raise RuntimeError(f"dual-sha mismatch src={src_sha} dst={dst_sha}")
    return src_sha, dst_sha


def classify_cleanup_residual(
    path: PathLike,
    allowed_exact: Set[str] | Iterable[str],
) -> str:
    """Classify a residual path against an exact allowlist.

    Returns ``\"ok\"`` if *path* (normalized string) is in *allowed_exact*,
    otherwise ``\"STOP_unknown\"``.
    """
    allowed = {str(item) for item in allowed_exact}
    candidates = {
        str(path),
        str(Path(path)),
        Path(path).as_posix(),
        os.path.normpath(str(path)),
    }
    if candidates & allowed:
        return "ok"
    # Also accept allowlist entries that match after normpath.
    allowed_norm = {os.path.normpath(a) for a in allowed}
    if os.path.normpath(str(path)) in allowed_norm:
        return "ok"
    return "STOP_unknown"


__all__ = [
    "ShortWriteError",
    "classify_cleanup_residual",
    "copy_file_o_excl",
    "write_bytes_o_excl",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: p1b_o_excl_copy.py SRC DST")
    src_sha, dst_sha = copy_file_o_excl(sys.argv[1], sys.argv[2])
    print(src_sha)
    assert src_sha == dst_sha
