"""R51.4 install: replace Gemma's L24 contribution with the student's prediction.

Install path chosen: monkey-patch `m._forward_layer`. Path A (forward hook on
`m.layers[target_layer]`) is not viable because `m.layers[i]` is a `GemmaLayer`
weight container, not an `nn.Module` with a `forward` — the actual forward
lives on the parent `GemmaSubstrate`. Path C (CardSlot) is residual-additive
and always runs *after* the target layer's full contribution, so it cannot
replace L24. Path B monkey-patches `m._forward_layer` so that, when called
with `layer_idx == target_layer`, we return `h_before + student(h_before)`
and skip Gemma's native L24 compute entirely.

KV-cache safety: Gemma 4 E4B has `n_layer_kv_from_start = 42 - 18 = 24`, so
L24 is the first shared-KV layer (`kv_source_layer(24, is_swa=...)` -> 22 or
23). L24 never owns its own KV cache, so skipping its forward does not leave
stale cache entries for downstream layers.

Uninstall: `handle.detach()` restores the original bound method on the same
instance. A follow-up `m.forward(prompt)` then matches pre-install output
exactly because no other state was touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InstallHandle:
    m: Any
    target_layer: int
    _original_forward_layer: Any
    _detached: bool = False

    def detach(self) -> None:
        if self._detached:
            return
        self.m._forward_layer = self._original_forward_layer
        self._detached = True


def install_r51_student(m, student, target_layer: int = 24) -> InstallHandle:
    """Route Gemma's L24 residual contribution through the trained student.

    On each `_forward_layer(h, layer, i, ...)` call with `i == target_layer`:
        h_before := h (residual entering L24)
        delta    := student(h_before)          [B, S, 2560]
        return      h_before + delta

    Other layers are forwarded unchanged. Installer grabs the currently-bound
    `m._forward_layer` (not the class-level function) so repeated install/
    detach cycles compose correctly even when other tooling also rebinds.
    """
    original = m._forward_layer

    device = next(student.parameters()).device if any(True for _ in student.parameters()) else torch.device("cpu")
    dtype = next(student.parameters()).dtype if any(True for _ in student.parameters()) else torch.float32

    def patched(h, layer, layer_idx, kv_cache=None, start_pos=0):
        if layer_idx != target_layer:
            return original(h, layer, layer_idx, kv_cache=kv_cache,
                            start_pos=start_pos)
        h_before = h
        x = h_before.to(device=device, dtype=dtype)
        with torch.no_grad():
            delta = student(x)
        delta = delta.to(device=h_before.device, dtype=h_before.dtype)
        return h_before + delta

    m._forward_layer = patched
    return InstallHandle(m=m, target_layer=target_layer,
                         _original_forward_layer=original)


def load_student_from_checkpoint(path: str, device: str = "cuda", eval_mode: bool = True):
    """Load an R51Student checkpoint and move it to `device`. Returns the
    `nn.Module` ready for `install_r51_student`.

    `weights_only=False` because the saved file contains the `R51StudentConfig`
    dataclass alongside the `state_dict` for self-describing checkpoints.
    """
    from calm.llm_computer.r51.student import R51Student, R51StudentConfig

    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        cfg = payload.get("config")
        if cfg is None:
            cfg = R51StudentConfig()
        elif isinstance(cfg, dict):
            cfg = R51StudentConfig(**cfg)
        student = R51Student(cfg)
        student.load_state_dict(payload["state_dict"])
    elif isinstance(payload, torch.nn.Module):
        student = payload
    else:
        cfg = R51StudentConfig()
        student = R51Student(cfg)
        student.load_state_dict(payload)

    student = student.to(device)
    if eval_mode:
        student.eval()
    return student
