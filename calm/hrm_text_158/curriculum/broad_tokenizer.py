"""Byte-level UTF-8 tokenizer for HRM-Text-1.58 Phase 3 broad curriculum.

Per codex msg 1779460698439 (Phase 3 Step 0 +1, route A1):

Deterministic vocab:
- ids 0-3 = specials: <pad>, <bos>, <eos>, <sep>
- ids 4-259 = byte values 0x00-0xff (`<byte:00>` .. `<byte:ff>`)

Vocab size = 260. NEVER built from corpus. Persisted as stable string list
in `vocab_as_list()`. Normalizer = identity (`byte_utf8_v1`).

Encoding: `text.encode("utf-8")` -> per-byte ids offset by 4.
Decoding: byte ids back to bytes then UTF-8 decode with `errors="replace"`.

Compatibility with Phase 2 GSM8k char tokenizer: NONE. Phase 3 ckpts
have different vocab size (260 vs 98) and different id-to-char mapping.
Phase 2 ckpts cannot be loaded into Phase 3 models. validate_load_from_ckpt_compat
in ckpt_compat.py hard-fails on this mismatch.
"""
from __future__ import annotations

from typing import Iterable


BROAD_NORMALIZER_VERSION = "byte_utf8_v1"

# Special tokens, ids 0-3
_SPECIALS = ("<pad>", "<bos>", "<eos>", "<sep>")
_BYTE_OFFSET = len(_SPECIALS)  # 4

# Full vocab size
VOCAB_SIZE = _BYTE_OFFSET + 256  # 260


class BroadTokenizer:
    """Byte-level UTF-8 char tokenizer for Phase 3.

    Identity normalizer (no semantic rewriting). Vocab is fixed at
    construction; not built from corpus.

    API mirrors Gsm8kTokenizer for trainer compatibility:
    - pad_id, bos_id, eos_id, sep_id
    - encode(text) -> list[int]
    - decode(ids) -> str
    - encode_example(question, target_int) -> (ids, sep_pos)
    - vocab_as_list() -> list[str]
    - assert_corpus_covered(rows) -> None
    - normalizer_version, vocab_size
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    sep_id = 3
    vocab_size = VOCAB_SIZE
    normalizer_version = BROAD_NORMALIZER_VERSION

    def __init__(self) -> None:
        # Vocab list: ids 0-3 specials, ids 4-259 byte placeholders
        self._vocab_list: list[str] = list(_SPECIALS) + [
            f"<byte:{b:02x}>" for b in range(256)
        ]

    # ---------- vocab API ---------- #

    def vocab_as_list(self) -> list[str]:
        """Ordered list[str] for persistence (id 0 first). Identical
        across all BroadTokenizer instances (deterministic by construction)."""
        return list(self._vocab_list)

    # ---------- encode / decode ---------- #

    def encode(self, text: str) -> list[int]:
        """Encode `text` (UTF-8) to byte token ids.

        text -> text.encode("utf-8") -> [byte_id + _BYTE_OFFSET for byte_id in bytes]

        Never raises (unlike Gsm8kTokenizer which raises on OOV); byte-level
        is OOV-free for any string.
        """
        return [b + _BYTE_OFFSET for b in text.encode("utf-8")]

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        """Decode token ids back to a string.

        Specials render as bracketed form (`<bos>`, `<sep>`, `<eos>`) except
        `<pad>` which renders as empty. With `stop_at_eos=True`, halts at
        first `<eos>`.

        Byte ids are batched into a `bytes` buffer then decoded as UTF-8
        with `errors="replace"` so partial multi-byte sequences become `?`
        rather than raising.
        """
        out_strs: list[str] = []
        byte_buf: list[int] = []

        def _flush_bytes():
            if byte_buf:
                out_strs.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in ids:
            if not isinstance(tid, int):
                tid = int(tid)
            if tid == self.eos_id and stop_at_eos:
                _flush_bytes()
                break
            if tid == self.pad_id:
                # skip; do not flush mid-byte
                continue
            if tid == self.bos_id:
                _flush_bytes()
                out_strs.append("<bos>")
                continue
            if tid == self.sep_id:
                _flush_bytes()
                out_strs.append("<sep>")
                continue
            if tid == self.eos_id:
                _flush_bytes()
                out_strs.append("<eos>")
                continue
            # Byte id
            if _BYTE_OFFSET <= tid < VOCAB_SIZE:
                byte_buf.append(tid - _BYTE_OFFSET)
            else:
                _flush_bytes()
                out_strs.append("?")
        _flush_bytes()
        return "".join(out_strs)

    # ---------- example construction ---------- #

    def encode_example(self, question: str, target_int: int) -> tuple[list[int], int]:
        """Build the `<bos> question <sep> {integer} <eos>` token sequence.

        Returns `(ids, sep_position)` matching GSM8k contract. sep_position
        is the index of `<sep>` in ids; training loss is masked to positions
        `>= sep_position` after left-shift (per shifted-PrefixLM contract).
        """
        q_ids = self.encode(question)
        t_ids = self.encode(str(target_int))
        ids = [self.bos_id] + q_ids + [self.sep_id] + t_ids + [self.eos_id]
        sep_pos = 1 + len(q_ids)
        return ids, sep_pos

    # ---------- corpus coverage gate ---------- #

    def assert_corpus_covered(self, rows: Iterable[dict], label: str = "corpus") -> None:
        """No-op for byte-level tokenizer.

        Byte-level UTF-8 is OOV-free by construction — any string encodes.
        Method present for trainer-API compatibility with Gsm8kTokenizer
        (which raises on OOV chars).
        """
        # Defensive sanity: iterate rows to surface any obvious data shape errors
        # but never raise on content.
        for r in rows:
            if not isinstance(r, dict):
                raise TypeError(
                    f"BroadTokenizer.assert_corpus_covered: row must be dict, "
                    f"got {type(r).__name__} in {label!r}"
                )
        return None
