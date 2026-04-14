"""Inference for Family A synth models — decode + functionally validate."""

from __future__ import annotations

from pathlib import Path

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.model import HRMConfig, HRMSeq2Seq
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, parse_expression
from calm.llm_computer.synth.data import SynthSample, encode_examples


class SynthFamilyAReasoner:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint missing: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.config = HRMConfig(**ckpt["config"])
        self.model = HRMSeq2Seq(self.config).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.val_acc = ckpt.get("val_acc", 0.0)

    def _encode_string(self, text: str) -> torch.Tensor:
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        ids = [bos] + [_CHAR_TO_ID[c] for c in text if c in _CHAR_TO_ID] + [eos]
        ids = ids[: self.config.max_seq_len]
        while len(ids) < self.config.max_seq_len:
            ids.append(pad)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def predict(self, sample: SynthSample) -> str:
        """Greedy-decode a predicted template string from the sample's IO pairs."""
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        enc = self._encode_string(encode_examples(sample))
        with torch.no_grad():
            mem = self.model.encode(enc)
            dec = [bos]
            for _ in range(self.config.max_dec_len - 1):
                padded = dec + [pad] * (self.config.max_dec_len - len(dec))
                dt = torch.tensor([padded], dtype=torch.long, device=self.device)
                logits = self.model.decode_step(dt, mem)
                nid = int(logits[0, len(dec) - 1, :].argmax().item())
                if nid == eos:
                    break
                dec.append(nid)
        out = ""
        for tid in dec[1:]:
            if tid in (pad, bos, eos):
                continue
            out += _ID_TO_CHAR.get(tid, "?")
        return out.strip()


def functional_correct(emitted: str, sample: SynthSample) -> bool:
    """Parse the emitted expression, substitute query (a, b), check it equals query_out."""
    # Substitute variable names to actual values.
    expr_concrete = emitted.replace("a", str(sample.query_a)).replace("b", str(sample.query_b))
    try:
        graph = parse_expression(expr_concrete)
        val = interpret(graph)
        if isinstance(val, float) and val == int(val):
            val = int(val)
        return val == sample.query_out
    except (ParseError, InterpreterError, ValueError):
        return False
