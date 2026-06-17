"""Typed sparse vote-event carrier for bounded-delta local update (r4b)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class SparseVoteEvents:
    """CPU sparse (index, vote) pairs without Python dict materialization on extraction."""

    indices: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.indices.dtype != torch.int64:
            raise ValueError(f"indices must be torch.int64, got {self.indices.dtype}")
        if self.values.dtype != torch.int16:
            raise ValueError(f"values must be torch.int16, got {self.values.dtype}")
        if tuple(self.indices.shape) != tuple(self.values.shape):
            raise ValueError("indices and values must have the same shape")
        if self.indices.dim() != 1:
            raise ValueError("indices and values must be 1-D tensors")
        if not self.indices.is_cpu or not self.values.is_cpu:
            raise ValueError("indices and values must be CPU tensors")
        if not self.indices.is_contiguous() or not self.values.is_contiguous():
            object.__setattr__(self, "indices", self.indices.contiguous())
            object.__setattr__(self, "values", self.values.contiguous())

    def validate(self, *, numel: int) -> None:
        if int(numel) <= 0:
            raise ValueError("numel must be > 0")
        if self.event_count() == 0:
            return
        if bool((self.indices < 0).any().item()) or bool((self.indices >= int(numel)).any().item()):
            raise ValueError("sparse vote indices out of range")
        if bool((self.values < -32768).any().item()) or bool((self.values > 32767).any().item()):
            raise ValueError("sparse vote values must fit int16")
        if bool((self.values == 0).any().item()):
            raise ValueError("sparse vote values must be non-zero")

    def event_count(self) -> int:
        return int(self.indices.numel())

    def to_dict(self) -> dict[int, int]:
        if self.event_count() == 0:
            return {}
        return {
            int(index): int(vote)
            for index, vote in zip(
                self.indices.tolist(),
                self.values.tolist(),
            )
        }

    @classmethod
    def from_dict(cls, events: Mapping[int, int]) -> SparseVoteEvents:
        items = [(int(index), int(vote)) for index, vote in events.items() if int(vote) != 0]
        if not items:
            return cls(
                indices=torch.empty(0, dtype=torch.int64),
                values=torch.empty(0, dtype=torch.int16),
            )
        items.sort(key=lambda item: item[0])
        indices = torch.tensor([index for index, _vote in items], dtype=torch.int64)
        values = torch.tensor([vote for _index, vote in items], dtype=torch.int16)
        return cls(indices=indices, values=values)

    @classmethod
    def from_dense_votes(cls, votes: torch.Tensor) -> SparseVoteEvents:
        flat = votes.detach().cpu().to(torch.int16).reshape(-1).contiguous()
        nz = torch.nonzero(flat != 0, as_tuple=False).flatten().to(torch.int64)
        if int(nz.numel()) == 0:
            return cls(
                indices=torch.empty(0, dtype=torch.int64),
                values=torch.empty(0, dtype=torch.int16),
            )
        return cls(indices=nz, values=flat.index_select(0, nz))
