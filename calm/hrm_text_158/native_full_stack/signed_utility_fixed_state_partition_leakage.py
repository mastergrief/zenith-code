"""Pure partition/leakage reducers for fixed-state signed-utility (D2c3 S1)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


class PartitionLeakageError(RuntimeError):
    pass


def _norm_text(s: Any) -> str:
    return " ".join(str(s).strip().lower().split())


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _row_sep(seps: Any, i: int) -> int:
    if seps is None:
        raise PartitionLeakageError("sep_positions_missing")
    v = seps[i]
    return int(v.item() if hasattr(v, "item") else v)


def _prompt_hash_from_tensors(inputs: Any, seps: Any, i: int) -> str:
    sep = _row_sep(seps, i)
    row = inputs[i].detach().cpu().contiguous()
    if sep < 0 or sep > int(row.shape[0]):
        raise PartitionLeakageError(f"sep_out_of_range:{sep}")
    prompt = row[:sep].contiguous()
    return _sha_bytes(f"{prompt.dtype}|{tuple(prompt.shape)}|".encode() + prompt.numpy().tobytes())


def _response_hash_from_labels(labels: Any, seps: Any, i: int) -> str:
    sep = _row_sep(seps, i)
    row = labels[i].detach().cpu().contiguous()
    resp = row[sep:].contiguous()
    return _sha_bytes(f"{resp.dtype}|{tuple(resp.shape)}|".encode() + resp.numpy().tobytes())


def surface_values(batch: Mapping[str, Any], surface: str) -> list[str]:
    """Derive leakage surface hashes; tensor prompts are strictly pre-sep."""
    meta = batch.get("metadata") or {}
    key = {
        "row_id": "row_ids",
        "normalized_prompt_hash": "normalized_prompt_hashes",
        "normalized_target_hash": "normalized_target_hashes",
        "response_token_hash": "response_token_hashes",
    }[surface]
    if key in meta and meta[key] is not None:
        return [str(x) for x in meta[key]]
    if surface == "row_id":
        raise PartitionLeakageError("leakage_row_ids_missing")
    tens = batch.get("batch") or {}
    if surface == "normalized_prompt_hash":
        if "prompts" in meta:
            return [_sha_text(_norm_text(p)) for p in meta["prompts"]]
        inputs, seps = tens.get("inputs"), tens.get("sep_positions")
        if inputs is None:
            raise PartitionLeakageError("leakage_surface_uncomputable:normalized_prompt_hash")
        return [_prompt_hash_from_tensors(inputs, seps, i) for i in range(int(inputs.shape[0]))]
    if surface == "normalized_target_hash":
        if "targets" in meta:
            return [_sha_text(_norm_text(t)) for t in meta["targets"]]
        labels, seps = tens.get("labels"), tens.get("sep_positions")
        if labels is None:
            raise PartitionLeakageError("leakage_surface_uncomputable:normalized_target_hash")
        return [_response_hash_from_labels(labels, seps, i) for i in range(int(labels.shape[0]))]
    if surface == "response_token_hash":
        if "response_tokens" in meta:
            return [_sha_text(json.dumps(list(r), separators=(",", ":"))) for r in meta["response_tokens"]]
        labels, seps = tens.get("labels"), tens.get("sep_positions")
        if labels is None:
            raise PartitionLeakageError("leakage_surface_uncomputable:response_token_hash")
        return [_response_hash_from_labels(labels, seps, i) for i in range(int(labels.shape[0]))]
    raise PartitionLeakageError(f"leakage_surface_uncomputable:{surface}")


def compute_partition_leakage_compact(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(batches) < 3:
        raise PartitionLeakageError("leakage_needs_three_batches")
    cap, ev = [batches[0]], [batches[1], batches[2]]
    out: dict[str, Any] = {}
    ok = True
    for surface, out_key in (
        ("row_id", "row_id_overlap"),
        ("normalized_prompt_hash", "normalized_prompt_hash_overlap"),
        ("normalized_target_hash", "normalized_target_hash_overlap"),
        ("response_token_hash", "response_token_hash_overlap"),
    ):
        a: set[str] = set()
        b: set[str] = set()
        for batch in cap:
            a.update(surface_values(batch, surface))
        for batch in ev:
            b.update(surface_values(batch, surface))
        n = len(a & b)
        out[out_key] = int(n)
        ok = ok and n == 0
    out["pass"] = bool(ok)
    return out


__all__ = [
    "PartitionLeakageError",
    "compute_partition_leakage_compact",
    "surface_values",
]
