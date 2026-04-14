"""
HRM Inference — load and run trained HRM models.

Usage:
    from calm.hrm.inference import HRMReasoner
    reasoner = HRMReasoner("calm/hrm/checkpoints/math_hrm_best.pt")
    answer = reasoner.reason("17 * 23 + 5")
    print(answer)  # "396"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from calm.hrm.model import HRM, HRMSeq2Seq, HRMConfig
from calm.hrm.data import (tokenize, detokenize, VOCAB_SIZE, _unreverse_if_numeric,
                            tokenize_trace, detokenize_trace)
from calm.expression import safe_eval, ExpressionError


class HRMReasoner:
    """Load and run inference on a trained HRM model."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Load model from checkpoint.

        Default device is CPU — 115K params runs sub-millisecond on CPU.
        No need to waste GPU VRAM.
        """
        self.device = device
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct config
        cfg = checkpoint["config"]
        self.config = HRMConfig(**cfg)

        # Load model
        self.model = HRM(self.config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.val_acc = checkpoint.get("val_acc", 0)
        self.epoch = checkpoint.get("epoch", 0)

    def reason(self, expression: str, max_new_tokens: int = 12) -> Optional[str]:
        """Run HRM reasoning on a math expression via autoregressive decoding.

        Seeds with `<bos>expr=`, then repeatedly runs the model and reads
        the next-token logit off the current last position. Stops on <eos>
        or after max_new_tokens.

        Args:
            expression: math expression like "17 * 23 + 5"
            max_new_tokens: cap on answer length in tokens

        Returns:
            Predicted answer string, or None if nothing was generated.
        """
        from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR

        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]

        expr_text = expression + "="
        ids = [bos_id] + [_CHAR_TO_ID[c] for c in expr_text if c in _CHAR_TO_ID]
        prompt_len = len(ids)  # answer tokens start here

        max_seq = self.config.max_seq_len
        for _ in range(max_new_tokens):
            if len(ids) >= max_seq:
                break
            # Pad current prefix to max_seq so model input shape is fixed.
            padded = ids + [pad_id] * (max_seq - len(ids))
            input_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
            # Next-token logits live at the position of the current last real token.
            next_id = int(logits[0, len(ids) - 1, :].argmax().item())
            if next_id == eos_id:
                break
            ids.append(next_id)

        chars = []
        for tid in ids[prompt_len:]:
            if tid in (pad_id, bos_id, eos_id):
                continue
            chars.append(_ID_TO_CHAR.get(tid, "?"))
        answer = "".join(chars).strip()
        return answer if answer else None

    def info(self) -> str:
        """Model info string."""
        return (f"HRM math: {self.model.param_count():,} params, "
                f"trained epoch {self.epoch}, val_acc={self.val_acc:.1%}")


class HRMSeq2SeqReasoner:
    """Load and run inference on a trained HRMSeq2Seq model.

    Encodes the prompt once (heavy, nested L/H recurrence), then decodes
    autoregressively from <bos> until <eos> or max_new_tokens. If the
    checkpoint was trained with `reverse_digits=True`, applies
    `_unreverse_if_numeric` before returning the string.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        cfg = checkpoint["config"]
        self.config = HRMConfig(**cfg)
        self.model = HRMSeq2Seq(self.config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.val_acc = checkpoint.get("val_acc", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.reverse_digits = checkpoint.get("reverse_digits", True)
        self.scratchpad = checkpoint.get("scratchpad", False)

    def reason(self, expression: str, max_new_tokens: int = 60) -> Optional[str]:
        """Encode the expression and autoregressively decode an answer.

        In scratchpad mode, intercepts `<end_call>` emissions: parses the
        expression between the last `<call>` and `<end_call>`, delegates to
        `safe_eval`, injects the result tokens as teacher-forced continuation,
        and resumes generation. Extracts the final answer from the completed
        trace (last number after last `=`).

        In answer-only mode, just runs the autoregressive loop and (optionally)
        reverses numeric digits on the way out.
        """
        from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR

        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]
        call_id = _CHAR_TO_ID["<call>"]
        end_call_id = _CHAR_TO_ID["<end_call>"]

        # Encoder input: <bos> expression <eos> pad...
        enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in expression if c in _CHAR_TO_ID] + [eos_id]
        enc_ids = enc_ids[: self.config.max_seq_len]
        while len(enc_ids) < self.config.max_seq_len:
            enc_ids.append(pad_id)
        enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            memory = self.model.encode(enc_tensor)

            dec_ids = [bos_id]
            max_dec = self.config.max_dec_len
            cap = min(max_new_tokens, max_dec - 1)
            steps = 0
            while steps < cap and len(dec_ids) < max_dec:
                padded = dec_ids + [pad_id] * (max_dec - len(dec_ids))
                dec_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
                logits = self.model.decode_step(dec_tensor, memory)
                next_id = int(logits[0, len(dec_ids) - 1, :].argmax().item())
                if next_id == eos_id:
                    break
                dec_ids.append(next_id)
                steps += 1

                # Scratchpad: intercept <end_call> → delegate to safe_eval, inject result.
                if self.scratchpad and next_id == end_call_id:
                    # Find the last <call> token
                    try:
                        call_pos = len(dec_ids) - 1 - dec_ids[::-1].index(call_id)
                    except ValueError:
                        continue  # <end_call> with no preceding <call>, skip
                    # Tokens strictly between <call> and <end_call> form the expression.
                    inner_ids = dec_ids[call_pos + 1 : len(dec_ids) - 1]
                    inner_str = detokenize_trace(inner_ids)
                    try:
                        result = safe_eval(inner_str)
                    except (ExpressionError, Exception):
                        continue
                    # Canonical stringification (int if float is whole).
                    if isinstance(result, bool):
                        result_str = str(result)
                    elif isinstance(result, float) and result == int(result):
                        result_str = str(int(result))
                    else:
                        result_str = str(result)
                    # Inject result tokens (teacher-forced continuation).
                    for c in result_str:
                        if c in _CHAR_TO_ID and len(dec_ids) < max_dec:
                            dec_ids.append(_CHAR_TO_ID[c])

        # Detokenize and extract final answer.
        if self.scratchpad:
            raw = detokenize_trace(dec_ids[1:])  # strip leading <bos>
            return _extract_final_answer(raw)

        # Answer-only mode (Round 1a behavior).
        chars = []
        for tid in dec_ids[1:]:
            if tid in (pad_id, bos_id, eos_id):
                continue
            chars.append(_ID_TO_CHAR.get(tid, "?"))
        raw = "".join(chars)
        if self.reverse_digits:
            raw = _unreverse_if_numeric(raw)
        raw = raw.strip()
        return raw if raw else None

    def info(self) -> str:
        mode = "scratchpad" if self.scratchpad else ("rev-digits" if self.reverse_digits else "plain")
        return (f"HRM-s2s math: {self.model.param_count():,} params, "
                f"trained epoch {self.epoch}, val_acc={self.val_acc:.1%}, mode={mode}")


def _extract_final_answer(trace_str: str) -> Optional[str]:
    """Pull the last number (or True/False) after the last `=` in a trace.

    Handles:
      "14 * 87 = 1218"                          → "1218"
      "a + b = c = 42"                          → "42"
      "<call>factorial(5)<end_call>120 = 120"  → "120"
      "<call>is_prime(17)<end_call>True = True" → "True"
    """
    if not trace_str:
        return None
    if "=" in trace_str:
        tail = trace_str.rsplit("=", 1)[1].strip()
    else:
        # No `=` — try the text after the last <end_call>.
        if "<end_call>" in trace_str:
            tail = trace_str.rsplit("<end_call>", 1)[1].strip()
        else:
            tail = trace_str.strip()
    # Take the first token (number, True, False).
    # Split on whitespace / semicolon; keep the leading token.
    for sep in (";", "\n"):
        if sep in tail:
            tail = tail.split(sep)[0].strip()
    # Already contains the answer as the leading chars; strip trailing junk.
    tail = tail.split()[0] if tail.split() else tail
    return tail if tail else None
