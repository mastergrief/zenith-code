"""Dispatcher — router HRM + sub-specialist HRMs with verified compute.

Round 3 of the L3-L6 roadmap (Layer 4: hierarchical routing). The
dispatcher classifies an incoming query via RouterHRM, then invokes
the matching sub-specialist HRM:

    query  →  RouterHRM  →  domain label  →  specialist HRM
                                                ↓
                                            math expression emit
                                                ↓
                                    extract_problem_from_trace
                                                ↓
                                        parse_expression
                                                ↓
                                           interpret
                                                ↓
                                        answer (verified)

If the label is `meta` and the query contains the <sep>-separated
3-example prefix, the meta-HRM handles it. If meta-HRM's checkpoint
isn't on disk yet (L3 still training), the dispatcher raises a clear
error for the caller.

This is the MINIMAL L4 in the roadmap — router over fixed specialists.
The full "spawn virtual specialist" version depends on L3's capacity
sweep converging; see RESEARCH_ROADMAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from calm.hrm.data import _CHAR_TO_ID
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.hrm.router_data import LABELS
from calm.hrm.router_model import RouterConfig, RouterHRM
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import (
    ParseError, extract_problem_from_trace, parse_expression,
)


DEFAULT_ROUTER_CKPT = "calm/hrm/checkpoints/router_best.pt"
DEFAULT_SPECIALIST_CKPTS: Dict[str, str] = {
    "math": "calm/hrm/checkpoints/math_structure_best.pt",
    "nl":   "calm/hrm/checkpoints/nl_math_structure_best.pt",
    "word": "calm/hrm/checkpoints/word_problem_best.pt",
    "gsm":  "calm/hrm/checkpoints/gsm_best.pt",
    "meta": "calm/hrm/checkpoints/meta_best.pt",
}


@dataclass
class DispatchResult:
    label: str
    emit: str                 # raw HRM decoder output
    expression: str           # extracted math expression
    answer: Optional[str]     # verified value, None if interpret failed


class Dispatcher:
    """Owns a RouterHRM + one HRMSeq2SeqReasoner per specialist (lazy-loaded)."""

    def __init__(self, router_ckpt: str = DEFAULT_ROUTER_CKPT,
                 specialist_ckpts: Optional[Dict[str, str]] = None,
                 device: str = "cpu"):
        self.device = device
        self.router_ckpt = router_ckpt
        self.specialist_ckpts = dict(specialist_ckpts or DEFAULT_SPECIALIST_CKPTS)
        self._router = None
        self._router_cfg = None
        self._specialists: Dict[str, HRMSeq2SeqReasoner] = {}

    def _load_router(self):
        if self._router is not None:
            return
        ckpt = torch.load(self.router_ckpt, map_location=self.device, weights_only=False)
        cfg = RouterConfig(**ckpt["config"])
        self._router_cfg = cfg
        self._router = RouterHRM(cfg).to(self.device)
        self._router.load_state_dict(ckpt["model_state_dict"])
        self._router.eval()

    def _load_specialist(self, label: str) -> HRMSeq2SeqReasoner:
        if label in self._specialists:
            return self._specialists[label]
        path = self.specialist_ckpts.get(label)
        if path is None:
            raise KeyError(f"no specialist checkpoint configured for label {label!r}")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"specialist checkpoint missing for label {label!r}: {path}")
        r = HRMSeq2SeqReasoner(path, device=self.device)
        self._specialists[label] = r
        return r

    def route(self, text_with_sep: str) -> str:
        """Classify `text_with_sep` and return the domain label.

        `text_with_sep` uses '\\x01' as the in-context separator marker (same
        convention as router_data.py). For raw queries with no prefix, just
        pass the plain text.
        """
        self._load_router()
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        sep = _CHAR_TO_ID["<sep>"]

        ids = [bos]
        for ch in text_with_sep:
            if ch == "\x01":
                ids.append(sep)
            elif ch in _CHAR_TO_ID:
                ids.append(_CHAR_TO_ID[ch])
        ids.append(eos)
        max_len = self._router_cfg.max_seq_len
        ids = ids[: max_len]
        while len(ids) < max_len:
            ids.append(pad)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = int(self._router(x).argmax(-1).item())
        return LABELS[pred]

    def _decode_through_specialist(self, reasoner: HRMSeq2SeqReasoner,
                                    text_with_sep: str) -> str:
        """Encode the text (with optional <sep> markers) and decode a trace."""
        from calm.hrm.data import _ID_TO_CHAR
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        sep = _CHAR_TO_ID["<sep>"]
        ids = [bos]
        for ch in text_with_sep:
            if ch == "\x01":
                ids.append(sep)
            elif ch in _CHAR_TO_ID:
                ids.append(_CHAR_TO_ID[ch])
        ids.append(eos)
        max_len = reasoner.config.max_seq_len
        ids = ids[: max_len]
        while len(ids) < max_len:
            ids.append(pad)
        enc_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            mem = reasoner.model.encode(enc_t)
            dec = [bos]
            for _ in range(reasoner.config.max_dec_len - 1):
                padded = dec + [pad] * (reasoner.config.max_dec_len - len(dec))
                dt = torch.tensor([padded], dtype=torch.long, device=self.device)
                logits = reasoner.model.decode_step(dt, mem)
                nid = int(logits[0, len(dec) - 1, :].argmax().item())
                if nid == eos:
                    break
                dec.append(nid)
        out = ""
        for tid in dec[1:]:
            if tid in (pad, bos, eos):
                continue
            out += _ID_TO_CHAR.get(tid, "?")
        return out

    def run(self, text_with_sep: str) -> DispatchResult:
        label = self.route(text_with_sep)
        reasoner = self._load_specialist(label)
        emit = self._decode_through_specialist(reasoner, text_with_sep)
        expr = extract_problem_from_trace(emit)
        try:
            val = interpret(parse_expression(expr))
            if isinstance(val, float) and val == int(val):
                val = int(val)
            answer = str(val)
        except (ParseError, InterpreterError):
            answer = None
        return DispatchResult(label=label, emit=emit, expression=expr, answer=answer)
