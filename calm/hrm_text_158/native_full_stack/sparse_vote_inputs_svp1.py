"""SVP1 binary sidecar encoding for sparse_vote_inputs_by_state_key."""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SVP1_MAGIC = b"SVP1"
SVP1_FORMAT_VERSION = 1
SVP1_ENDIAN_TAG = 0x4544  # little-endian marker ('ED')
SPARSE_VOTE_PAIRS_ENCODING = "sparse_vote_pairs_v1"
SPARSE_VOTE_SIDECAR_SUFFIX = ".svp1"
MAX_SVP1_INDEX = 2**31 - 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inline_sparse_vote_inputs_by_state_key(
    votes_by_key: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, int]]:
    """Reference inline dict encoding (backward-compat + equivalence oracle)."""
    sparse: dict[str, dict[str, int]] = {}
    for state_key in sorted(votes_by_key):
        vote_flat = votes_by_key[state_key].detach().cpu().flatten()
        vote_nz = torch.nonzero(vote_flat != 0, as_tuple=False).flatten()
        if vote_nz.numel() == 0:
            sparse[str(state_key)] = {}
            continue
        values = vote_flat[vote_nz]
        indices = vote_nz.numpy()
        values_np = values.numpy()
        sparse[str(state_key)] = dict(
            zip(
                map(str, indices.tolist()),
                map(int, values_np.tolist()),
            )
        )
    return sparse


def inline_sparse_votes_from_record(raw: Mapping[str, Any]) -> dict[str, dict[int, int]]:
    votes_by_key: dict[str, dict[int, int]] = {}
    for state_key in sorted(raw):
        lane_map = {
            int(flat_index): int(vote)
            for flat_index, vote in dict(raw[state_key]).items()
            if int(vote) != 0
        }
        votes_by_key[str(state_key)] = lane_map
    return votes_by_key


def encode_sparse_vote_inputs_svp1(
    votes_by_key: Mapping[str, torch.Tensor],
) -> tuple[bytes, dict[str, int], int]:
    """Encode nonzero votes to SVP1 bytes + per-state nnz counts."""
    chunks: list[bytes] = [
        SVP1_MAGIC,
        struct.pack("<HH", int(SVP1_FORMAT_VERSION), int(SVP1_ENDIAN_TAG)),
    ]
    per_state: dict[str, int] = {}
    total = 0
    for state_key in sorted(votes_by_key):
        key_bytes = str(state_key).encode("utf-8")
        if not key_bytes:
            raise ValueError("state_key must be non-empty UTF-8")
        vote_flat = votes_by_key[state_key].detach().cpu().flatten()
        vote_nz = torch.nonzero(vote_flat != 0, as_tuple=False).flatten()
        nnz = int(vote_nz.numel())
        per_state[str(state_key)] = nnz
        total += nnz
        chunks.append(struct.pack("<I", len(key_bytes)))
        chunks.append(key_bytes)
        chunks.append(struct.pack("<I", nnz))
        if nnz == 0:
            continue
        indices = vote_nz.numpy().astype(np.int64, copy=False)
        values = vote_flat[vote_nz].numpy()
        if indices.size:
            if (indices < 0).any():
                raise ValueError("flat index out of range for SVP1")
            if int(indices.max()) > MAX_SVP1_INDEX:
                raise ValueError("flat index >= 2^31 is forbidden on SVP1 path")
        index_array = indices.astype(np.int32, copy=False)
        value_array = values.astype(np.int16, copy=False)
        if not np.array_equal(index_array.astype(np.int64), indices):
            raise ValueError("flat index would overflow int32 encoding")
        if not np.array_equal(value_array.astype(np.int32), values.astype(np.int32)):
            raise ValueError("vote value would truncate when encoded as int16")
        chunks.append(index_array.tobytes(order="C"))
        chunks.append(value_array.tobytes(order="C"))
    return b"".join(chunks), per_state, total


def build_sparse_vote_inputs_stub(
    *,
    step_name: str,
    per_state: Mapping[str, int],
    total: int,
    sidecar_sha256: str,
) -> dict[str, Any]:
    return {
        "encoding": SPARSE_VOTE_PAIRS_ENCODING,
        "format_version": int(SVP1_FORMAT_VERSION),
        "endian": "little",
        "sidecar_relpath": f"per_step/{step_name}_sparse_votes{SPARSE_VOTE_SIDECAR_SUFFIX}",
        "sidecar_sha256": str(sidecar_sha256),
        "nonzero_entry_count": int(total),
        "per_state": {str(key): int(value) for key, value in sorted(per_state.items())},
    }


def validate_sidecar_relpath(sidecar_relpath: str) -> None:
    rel = str(sidecar_relpath)
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        raise ValueError("sidecar_relpath must be relative")
    if ".." in Path(rel).parts:
        raise ValueError("sidecar_relpath must not contain ..")
    if not rel.endswith(SPARSE_VOTE_SIDECAR_SUFFIX):
        raise ValueError(f"sidecar_relpath must end with {SPARSE_VOTE_SIDECAR_SUFFIX}")


def resolve_sidecar_path(votes_emit_root: Path, sidecar_relpath: str) -> Path:
    validate_sidecar_relpath(sidecar_relpath)
    root = Path(votes_emit_root).resolve()
    candidate = (root / sidecar_relpath).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("sidecar_relpath escapes votes_emit root") from exc
    if candidate.is_symlink():
        raise ValueError("sidecar path must not be a symlink")
    return candidate


def decode_sparse_vote_inputs_svp1(sidecar_bytes: bytes) -> tuple[dict[str, dict[int, int]], dict[str, int], int]:
    offset = 0
    if len(sidecar_bytes) < 8:
        raise ValueError("truncated SVP1 sidecar")
    if sidecar_bytes[offset : offset + 4] != SVP1_MAGIC:
        raise ValueError("invalid SVP1 magic")
    offset += 4
    format_version, endian_tag = struct.unpack_from("<HH", sidecar_bytes, offset)
    offset += 4
    if int(format_version) != int(SVP1_FORMAT_VERSION):
        raise ValueError(f"unsupported SVP1 format_version: {format_version}")
    if int(endian_tag) != int(SVP1_ENDIAN_TAG):
        raise ValueError(f"unsupported SVP1 endian tag: {endian_tag}")

    votes_by_key: dict[str, dict[int, int]] = {}
    per_state: dict[str, int] = {}
    total = 0
    seen_state_keys: set[str] = set()
    prior_state_key: str | None = None
    while offset < len(sidecar_bytes):
        if offset + 4 > len(sidecar_bytes):
            raise ValueError("truncated SVP1 state header")
        key_len = struct.unpack_from("<I", sidecar_bytes, offset)[0]
        offset += 4
        if offset + key_len + 4 > len(sidecar_bytes):
            raise ValueError("truncated SVP1 state key")
        key_bytes = sidecar_bytes[offset : offset + key_len]
        offset += key_len
        try:
            state_key = key_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("state_key is not valid UTF-8") from exc
        if state_key in seen_state_keys:
            raise ValueError(f"duplicate state_key in SVP1 sidecar: {state_key}")
        if prior_state_key is not None and state_key < prior_state_key:
            raise ValueError("state keys must be sorted lexicographically")
        seen_state_keys.add(state_key)
        prior_state_key = state_key

        nnz = struct.unpack_from("<I", sidecar_bytes, offset)[0]
        offset += 4
        per_state[state_key] = int(nnz)
        total += int(nnz)
        lane_map: dict[int, int] = {}
        if nnz:
            index_bytes = nnz * 4
            value_bytes = nnz * 2
            if offset + index_bytes + value_bytes > len(sidecar_bytes):
                raise ValueError("truncated SVP1 payload")
            indices = np.frombuffer(
                sidecar_bytes,
                dtype="<i4",
                count=nnz,
                offset=offset,
            )
            offset += index_bytes
            values = np.frombuffer(
                sidecar_bytes,
                dtype="<i2",
                count=nnz,
                offset=offset,
            )
            offset += value_bytes
            if (indices < 0).any():
                raise ValueError("flat index out of range for SVP1")
            if indices.size and int(indices.max()) > MAX_SVP1_INDEX:
                raise ValueError("flat index >= 2^31 is forbidden on SVP1 path")
            if (values < np.iinfo(np.int16).min).any() or (
                values > np.iinfo(np.int16).max
            ).any():
                raise ValueError("vote value not int16-representable")
            if (values == 0).any():
                raise ValueError("zero vote must not appear in sparse sidecar")
            if indices.size > 1 and not np.all(np.diff(indices) > 0):
                raise ValueError("indices must be strictly ascending")
            lane_map = dict(zip(indices.tolist(), values.tolist(), strict=True))
        votes_by_key[state_key] = lane_map

    if offset != len(sidecar_bytes):
        raise ValueError("trailing bytes in SVP1 sidecar")
    return votes_by_key, per_state, total


def write_sidecar_atomically(target_path: Path, sidecar_bytes: bytes) -> None:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(sidecar_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target_path)


def verify_sparse_vote_inputs_stub(
    sparse_field: Mapping[str, Any],
    *,
    votes_emit_root: Path,
    emit_path: Path | None = None,
) -> dict[str, dict[int, int]]:
    if not isinstance(sparse_field, Mapping):
        raise ValueError("sparse_vote_inputs_by_state_key must be a mapping")
    encoding = sparse_field.get("encoding")
    if encoding is None:
        return inline_sparse_votes_from_record(sparse_field)
    if str(encoding) != SPARSE_VOTE_PAIRS_ENCODING:
        raise ValueError(f"unsupported sparse_vote_inputs encoding: {encoding}")

    sidecar_relpath = str(sparse_field.get("sidecar_relpath", ""))
    expected_sha = str(sparse_field.get("sidecar_sha256", ""))
    expected_total = sparse_field.get("nonzero_entry_count")
    per_state_raw = sparse_field.get("per_state")
    if not sidecar_relpath or not expected_sha:
        raise ValueError("SVP1 stub missing sidecar_relpath or sidecar_sha256")
    if not isinstance(per_state_raw, Mapping):
        raise ValueError("SVP1 stub missing per_state mapping")
    if expected_total is None:
        raise ValueError("SVP1 stub missing nonzero_entry_count")

    sidecar_path = resolve_sidecar_path(votes_emit_root, sidecar_relpath)
    if emit_path is not None:
        expected_name = Path(sidecar_relpath).name
        if sidecar_path.name != expected_name:
            raise ValueError("resolved sidecar filename mismatch")
    if not sidecar_path.is_file():
        raise ValueError(f"missing SVP1 sidecar: {sidecar_path}")
    sidecar_bytes = sidecar_path.read_bytes()
    actual_sha = _sha256_bytes(sidecar_bytes)
    if actual_sha != expected_sha:
        raise ValueError("SVP1 sidecar sha256 mismatch")

    decoded, decoded_per_state, decoded_total = decode_sparse_vote_inputs_svp1(sidecar_bytes)
    per_state = {str(key): int(value) for key, value in per_state_raw.items()}
    if int(expected_total) != int(decoded_total):
        raise ValueError("SVP1 nonzero_entry_count mismatch")
    if per_state != decoded_per_state:
        raise ValueError("SVP1 per_state nnz mismatch")
    if sum(per_state.values()) != int(decoded_total):
        raise ValueError("SVP1 per_state counts do not sum to nonzero_entry_count")
    return decoded


def sparse_votes_from_emit_record(
    record: Mapping[str, Any],
    *,
    votes_emit_root: Path | None = None,
    emit_path: Path | None = None,
) -> dict[str, dict[int, int]]:
    raw = record.get("sparse_vote_inputs_by_state_key", {})
    if votes_emit_root is None:
        if isinstance(raw, Mapping) and raw.get("encoding") == SPARSE_VOTE_PAIRS_ENCODING:
            raise ValueError("SVP1 sparse_vote_inputs requires votes_emit_root")
        return inline_sparse_votes_from_record(raw)
    return verify_sparse_vote_inputs_stub(
        raw,
        votes_emit_root=Path(votes_emit_root),
        emit_path=emit_path,
    )
