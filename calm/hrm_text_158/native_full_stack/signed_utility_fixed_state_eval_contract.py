"""Deterministic + autocast-off eval contract (D2c3 S3)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def deterministic_eval_contract(*, device: str | None = None) -> Iterator[None]:
    import torch

    device_type = "cpu"
    if device is not None:
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    elif torch.cuda.is_available():
        device_type = "cuda"
    prev = {
        "det": torch.are_deterministic_algorithms_enabled(),
        "cudnn_det": bool(torch.backends.cudnn.deterministic),
        "cudnn_bench": bool(torch.backends.cudnn.benchmark),
    }
    tf32: dict[str, bool] = {}
    if torch.cuda.is_available():
        tf32["mat"] = bool(torch.backends.cuda.matmul.allow_tf32)
        tf32["cdn"] = bool(torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(prev["det"])
        torch.backends.cudnn.deterministic = prev["cudnn_det"]
        torch.backends.cudnn.benchmark = prev["cudnn_bench"]
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32["mat"]
            torch.backends.cudnn.allow_tf32 = tf32["cdn"]


__all__ = ["deterministic_eval_contract"]
