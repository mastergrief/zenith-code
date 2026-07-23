"""Model/runtime setup for forgetting-mechanism screen (r6c).

Owns: dynamic repo import + file hashing, checkpoint/hash/model construction,
eligible-module discovery + FixedQScale forward install, loss/decode primitives.

Does NOT own train-state mutation or receipt JSON emission.
Bound by PLAN_v9 sha 07a02aff… + authority 1784812148229.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from typing import Any

import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import (
    begin_credit_step,
    fixed_qscale_linear_with_credit,
    get_credit_store,
)

BULK_SUFFIXES = (
    "gqkv_proj.weight",
    "o_proj.weight",
    "gate_up_proj.weight",
    "down_proj.weight",
)
EXPECTED_PARENT_SHA256 = (
    "2d9b9f6746e66cec9e7e39d65e8171151e836daca99df6b56fb488d8a6f2403b"
)


def _scripts_dir() -> str:
    """Repo scripts/ from this module (native_full_stack/ -> ../../../scripts)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "scripts"))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _load_probe_and_screen():
    base = _scripts_dir()
    out = []
    for name in ("probe_hrm_text_158", "hrm_text_158_rotor_backward_saved_screen"):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(base, name + ".py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(mod)
    return out


def _eligible_param_names(model) -> list[str]:
    return [n for n, _p in model.named_parameters() if n.endswith(BULK_SUFFIXES)]


def _param_to_module(model) -> dict[str, BitLinear]:
    out: dict[str, BitLinear] = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            wname = f"{mod_name}.weight" if mod_name else "weight"
            if wname.endswith(BULK_SUFFIXES):
                out[wname] = mod
    return out


def _install_fixed_qscale_forwards(
    modules: dict[str, BitLinear],
    q_levels: dict[str, torch.Tensor],
    frozen_scales: dict[str, torch.Tensor],
) -> dict[str, Any]:
    import types

    originals: dict[str, Any] = {}
    for pname, mod in modules.items():
        originals[pname] = mod.forward

        def _fwd(self, x, _pname=pname):
            q = q_levels[_pname].to(device=x.device)
            s = frozen_scales[_pname].to(device=x.device)
            return fixed_qscale_linear_with_credit(
                x, q, s, self.bias, name=_pname
            )

        mod.forward = types.MethodType(_fwd, mod)  # type: ignore[method-assign]
        orig_qw = mod.quantize_weight

        def _qw(self, _orig=orig_qw):
            get_credit_store().n_bitlinear_dynamic_forwards += 1
            return _orig()

        mod.quantize_weight = types.MethodType(_qw, mod)  # type: ignore[method-assign]
    return originals


def rebind_fixed_qscale_forwards(
    modules: dict[str, BitLinear],
    q_levels: dict[str, torch.Tensor],
    frozen_scales: dict[str, torch.Tensor],
) -> None:
    """Re-install FixedQScale closures bound to the PROVIDED q_levels dict.

    Call after any q_levels dict replacement so model.forward mutates/reads the
    same object the train loop updates. Prefer mutating the original dict in
    place; use this when a fresh dict is unavoidable.
    """
    _install_fixed_qscale_forwards(modules, q_levels, frozen_scales)


def assert_q_levels_coupled(
    rt: dict[str, Any],
    q_levels: dict[str, torch.Tensor],
) -> None:
    """Fail-closed: train-loop q_levels must be the EXACT object captured by forwards."""
    if q_levels is not rt.get("q_levels"):
        raise RuntimeError(
            "q-forward decoupling: train q_levels is not rt['q_levels'] "
            "(clone disconnected from installed FixedQScale closures)"
        )


def _loss_and_credit(m, tok, rows, *, max_seq_len: int, device: str, eligible: list[str]):
    from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID

    begin_credit_step(eligible)
    m.zero_grad(set_to_none=True)
    total_loss = None
    for r in rows:
        q_ids = tok.encode(r["question"])
        a_ids = tok.encode(str(r["expected"])) + [tok.eos_id]
        seq = [tok.bos_id] + q_ids + [tok.sep_id] + a_ids
        if len(seq) > max_seq_len:
            continue
        sep_pos = 1 + len(q_ids)
        labels = [IGNORE_LABEL_ID] * (sep_pos + 1) + a_ids
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        lab = torch.tensor([labels], dtype=torch.long, device=device)
        sep_t = torch.tensor([sep_pos], dtype=torch.long, device=device)
        pos = torch.arange(ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        _carry, loss, _metrics = m(
            None,
            {"inputs": ids, "labels": lab, "sep_positions": sep_t, "position_ids": pos},
        )
        total_loss = loss if total_loss is None else total_loss + loss
    if total_loss is None:
        raise RuntimeError("no rows produced a loss")
    total_loss.backward()
    store = get_credit_store()
    store.assert_route_completeness(eligible)
    grads = store.snapshot_and_mark()
    return float(total_loss.item()), grads, store


def _decode_greedy(m, tok, question: str, *, max_gen: int, max_seq_len: int, device: str):
    """Probe exact-match decode under installed FixedQScale forwards."""
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    if len(prefix) >= max_seq_len:
        return "", True
    sep_pos = 1 + len(q_ids)
    out_tokens: list[int] = []
    cur = list(prefix)
    # Credit store must be begun for FixedQScale forward; no backward on probes.
    begin_credit_step([])
    for _ in range(max_gen):
        if len(cur) >= max_seq_len:
            break
        ids = torch.tensor([cur], dtype=torch.long, device=device)
        sep_t = torch.tensor([sep_pos], dtype=torch.long, device=device)
        pos = torch.arange(ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            _carry, logits = m(
                None,
                {"inputs": ids, "sep_positions": sep_t, "position_ids": pos},
            )
        if not bool(torch.isfinite(logits).all().item()):
            break
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id == tok.eos_id:
            break
        out_tokens.append(next_id)
        cur.append(next_id)
    return tok.decode(out_tokens, stop_at_eos=False), False


def _exact_match_count(
    m, tok, rows, *, max_seq_len: int, device: str, max_gen: int = 8
) -> int:
    n_ok = 0
    for q, e, _sr in rows:
        decoded, too_long = _decode_greedy(
            m, tok, q, max_gen=max_gen, max_seq_len=max_seq_len, device=device
        )
        if (not too_long) and decoded == str(e):
            n_ok += 1
    return n_ok


def load_and_patch_runtime(
    *,
    ckpt_path: str,
    device: str,
) -> dict[str, Any]:
    """Load banked parent, derive q/scale, install FixedQScale forwards.

    Returns a runtime dict consumed by the execution loop / receipt assembly.
    Does not mutate train-state (acc/episode); only installs forward patches.
    """
    sha_before = _sha256_file(ckpt_path)
    if sha_before != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"banked sha mismatch before load: got {sha_before}, "
            f"expected {EXPECTED_PARENT_SHA256}"
        )

    torch.set_num_threads(8)
    probe, _scr = _load_probe_and_screen()

    print(f"[forget-mech] loading ckpt ({device}): {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    max_seq_len = int(ckpt["config"]["max_seq_len"])
    m, tok = probe._build_model_from_ckpt(ckpt, device)
    m.train()

    modules = _param_to_module(m)
    eligible = list(modules.keys())
    if not eligible:
        raise SystemExit("no eligible BitLinear bulk projections found")

    q_levels: dict[str, torch.Tensor] = {}
    frozen_scales: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, mod in modules.items():
            p = mod.weight
            scale = p.detach().float().abs().mean().clamp(min=1e-5)
            frozen_scales[n] = scale.detach().cpu().to(torch.float32).reshape(())
            q_levels[n] = (
                (p.detach().float() / scale).round().clamp(-1, 1).to(torch.int8).cpu()
            )

    scale_sha_before = hashlib.sha256(
        b"".join(_sha_tensor(frozen_scales[n]).encode() for n in sorted(frozen_scales))
    ).hexdigest()
    q_sha_before = hashlib.sha256(
        b"".join(_sha_tensor(q_levels[n]).encode() for n in sorted(q_levels))
    ).hexdigest()

    _install_fixed_qscale_forwards(modules, q_levels, frozen_scales)

    return {
        "m": m,
        "tok": tok,
        "modules": modules,
        "eligible": eligible,
        "q_levels": q_levels,
        "frozen_scales": frozen_scales,
        "max_seq_len": max_seq_len,
        "sha_before": sha_before,
        "scale_sha_before": scale_sha_before,
        "q_sha_before": q_sha_before,
        "ckpt_path": ckpt_path,
        "device": device,
    }
