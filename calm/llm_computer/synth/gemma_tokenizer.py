"""Minimal Gemma tokenizer extracted from GGUF vocabulary.

SentencePiece-compatible encode/decode using the vocab table from
the GGUF file. Not a full SentencePiece implementation — uses greedy
longest-prefix matching for encoding. Good enough for inference demos;
for production, use the real SentencePiece model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class GemmaTokenizer:
    """Simple greedy tokenizer from GGUF vocab."""

    BOS_ID = 2   # <bos> in Gemma 4 GGUF (note: swapped vs some models)
    EOS_ID = 1   # <eos>
    PAD_ID = 0
    # Gemma uses ▁ (U+2581) as word separator in SentencePiece
    SP_PREFIX = "▁"

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.id_to_token = {i: t for i, t in enumerate(vocab)}
        # Build token → id lookup (longest match first)
        self.token_to_id = {}
        for i, t in enumerate(vocab):
            if t not in self.token_to_id:  # first occurrence wins
                self.token_to_id[t] = i
        # Sort tokens by length desc for greedy matching
        self._sorted_tokens = sorted(
            [(t, i) for t, i in self.token_to_id.items()
             if len(t) > 0 and not t.startswith("<") and t != "\t\x00\x00\x00"],
            key=lambda x: len(x[0]), reverse=True
        )

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "GemmaTokenizer":
        """Extract vocabulary from GGUF file."""
        from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf
        reader = read_turboquant_gguf(gguf_path)
        tokens_field = reader.fields["tokenizer.ggml.tokens"]
        vocab = []
        parts = tokens_field.parts
        i = 1
        while i + 1 < len(parts) and len(vocab) < 262146:  # over-read slightly
            try:
                data_part = parts[i + 1]
                token_str = bytes(data_part).decode("utf-8", errors="replace")
                vocab.append(token_str)
                i += 2
            except Exception:
                vocab.append("")
                i += 2
        # The GGUF string array has 2 metadata entries at the start that
        # get parsed as garbled tokens. Skip them to align with token IDs.
        if len(vocab) > 2 and vocab[0].startswith("\t") and vocab[2] == "<pad>":
            vocab = vocab[2:]  # align: token ID 0 = <pad>, 1 = <eos>, 2 = <bos>
        vocab = vocab[:262144]
        # Fix BOS/EOS based on GGUF metadata
        bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
        eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
        tok = cls(vocab)
        if bos_field:
            tok.BOS_ID = int(bos_field.parts[-1][0])
        if eos_field:
            tok.EOS_ID = int(eos_field.parts[-1][0])
        return tok

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Encode text to token IDs using greedy longest-prefix matching."""
        # SentencePiece convention: prepend ▁ to represent word boundary
        text = self.SP_PREFIX + text.replace(" ", self.SP_PREFIX)

        ids = []
        if add_bos:
            ids.append(self.BOS_ID)

        pos = 0
        while pos < len(text):
            matched = False
            for token, tid in self._sorted_tokens:
                if text[pos:pos + len(token)] == token:
                    ids.append(tid)
                    pos += len(token)
                    matched = True
                    break
            if not matched:
                # Byte fallback: encode as individual byte tokens
                byte_val = text[pos].encode("utf-8")
                for b in byte_val:
                    # Gemma byte tokens are at specific positions
                    # Try common byte-fallback range
                    byte_token = f"<0x{b:02X}>"
                    if byte_token in self.token_to_id:
                        ids.append(self.token_to_id[byte_token])
                    else:
                        ids.append(3)  # unknown
                pos += 1

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for tid in ids:
            if tid in (self.BOS_ID, self.PAD_ID):
                continue
            if tid == self.EOS_ID:
                break
            t = self.id_to_token.get(tid, "")
            tokens.append(t)
        text = "".join(tokens)
        # Remove SentencePiece word separators
        text = text.replace(self.SP_PREFIX, " ")
        if text.startswith(" "):
            text = text[1:]
        return text
