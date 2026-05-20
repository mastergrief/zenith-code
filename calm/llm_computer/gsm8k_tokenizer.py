"""Char-level tokenizer for real GSM8k word problems.

Locked S0b2 contract per ai-room audit thread `1779313584790-456ee5a3`
and prior `1779313349390-1fbcba07`:

- Local 98-token vocab (4 special + 94 declared chars) built from
  train+val only — the test split is an OOV check, NOT a vocab source.
- Normalizer applied to question + target before tokenize. Version
  pinned by `NORMALIZER_VERSION`; bump on any rule change so old
  checkpoints fail loudly on incompatible data.
- Reserved-extras included in the declared vocab regardless of train+val
  occurrence (a-priori justified standard punctuation).
- Target format: `<bos> question <sep> {integer} <eos>` (final-integer-only).
- Hard-fail at startup if any corpus char is OOV vs the declared vocab.
"""
from __future__ import annotations

from typing import Iterable


NORMALIZER_VERSION = "v2"

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>")

# Reserved-extras chars: included in the declared vocab regardless of
# train+val presence. Justification must be a-priori (standard punctuation
# / common real-world usage), NOT "test set has it." Adding any char here
# bumps NORMALIZER_VERSION.
#
# v2 additions:
#   - '#' — number/identifier prefix (e.g. "model #2"). The post-normalize
#     OOV check on the test split proves coverage; do not cite specific
#     test rows here.
_RESERVED_EXTRAS = frozenset({"#"})

_SMART_QUOTE = {
    "‘": "'",   # left single
    "’": "'",   # right single / apostrophe
    "“": '"',   # left double
    "”": '"',   # right double
}
_SMART_DASH = {
    "–": "-",   # en dash
    "—": "-",   # em dash
}
_WHITESPACE_NORMALIZE = {
    # Use \u escapes — source-file rendering of literal unicode
    # chars is not byte-stable across editor/tooling roundtrips;
    # getting `\u00a0` (NBSP) silently rewritten to U+0020 would
    # map all spaces to newlines.
    "\u00a0": " ",   # non-breaking space → space
    "\u200b": "",    # zero-width space → drop
    "\u2028": "\n",  # line separator → newline
    "\u2029": "\n",  # paragraph separator → newline
}


def normalize_text(s: str) -> str:
    """Normalizer (version pinned by `NORMALIZER_VERSION`) — applied to
    question + target before tokenize.

    Retained chars (load-bearing for word-problem semantics):
        $ % ? : & ' " ^ # ! ¢ £ €

    Mapped:
        smart quotes ‘ ’ → '   “ ” → "
        en/em dashes – — → -
        non-breaking space → space
        zero-width space → drop
        line separator → newline
    """
    out_chars: list[str] = []
    for ch in s:
        if ch in _WHITESPACE_NORMALIZE:
            out_chars.append(_WHITESPACE_NORMALIZE[ch])
        elif ch in _SMART_QUOTE:
            out_chars.append(_SMART_QUOTE[ch])
        elif ch in _SMART_DASH:
            out_chars.append(_SMART_DASH[ch])
        else:
            out_chars.append(ch)
    return "".join(out_chars)


class Gsm8kTokenizer:
    """Char-level tokenizer with hard-fail on OOV.

    Build paths:
      - `from_corpus(train_val_rows)` — derive vocab from train+val only.
      - `from_metadata(vocab_list, normalizer_version)` — reload from a
        saved checkpoint's `gsm8k_char_vocab` + `gsm8k_normalizer_version`.

    Mismatched `normalizer_version` at reload raises immediately.
    """

    def __init__(self, char_to_id: dict[str, int],
                 normalizer_version: str = NORMALIZER_VERSION):
        if normalizer_version != NORMALIZER_VERSION:
            raise ValueError(
                f"tokenizer normalizer_version={normalizer_version!r} "
                f"!= module NORMALIZER_VERSION={NORMALIZER_VERSION!r}; "
                f"data preprocessing has changed since the checkpoint was "
                f"saved. Reproduce the old preprocessing or retrain."
            )
        self.char_to_id = dict(char_to_id)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.pad_id = self.char_to_id["<pad>"]
        self.bos_id = self.char_to_id["<bos>"]
        self.eos_id = self.char_to_id["<eos>"]
        self.sep_id = self.char_to_id["<sep>"]
        self.normalizer_version = normalizer_version

    @classmethod
    def from_corpus(cls, train_val_rows: Iterable[dict]) -> "Gsm8kTokenizer":
        chars: set[str] = set()
        for r in train_val_rows:
            chars.update(normalize_text(r["question"]))
            chars.update(normalize_text(str(r["expected"])))
        chars |= _RESERVED_EXTRAS
        # Special tokens get the lowest ids (pad=0); declared chars sorted
        # for deterministic vocab order across runs.
        char_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for c in sorted(chars):
            char_to_id[c] = len(char_to_id)
        return cls(char_to_id)

    @classmethod
    def from_metadata(cls, vocab_list: list[str],
                      normalizer_version: str) -> "Gsm8kTokenizer":
        """Reload from checkpoint metadata. `vocab_list[i]` is the token
        whose id is `i`.
        """
        char_to_id = {tok: i for i, tok in enumerate(vocab_list)}
        return cls(char_to_id, normalizer_version=normalizer_version)

    def vocab_as_list(self) -> list[str]:
        """Ordered list[str] for persistence (id 0 first)."""
        return [self.id_to_char[i] for i in range(self.vocab_size)]

    def encode(self, text: str) -> list[int]:
        """Encode `text` (post-normalize) to a list of token ids.

        Raises `KeyError` on any char not in vocab — callers must run
        `assert_corpus_covered` at startup to surface OOV early.
        """
        return [self.char_to_id[c] for c in normalize_text(text)]

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        """Decode a list of token ids back to a string. Special tokens
        render as their bracketed form (`<bos>`, `<sep>`, etc.) except
        `<pad>` which renders as empty.

        With `stop_at_eos=True` (default), decoding halts at the first
        `<eos>` token encountered.
        """
        out: list[str] = []
        for tid in ids:
            tok = self.id_to_char.get(tid, "?")
            if stop_at_eos and tok == "<eos>":
                break
            if tok == "<pad>":
                continue
            out.append(tok)
        return "".join(out)

    def encode_example(self, question: str,
                       target_int: int) -> tuple[list[int], int]:
        """Build the locked `<bos> question <sep> {integer} <eos>` token
        sequence.

        Returns `(ids, sep_position)` — `sep_position` is the index of
        the `<sep>` token. Training loss is masked to positions
        `> sep_position` (the target chars + `<eos>`).
        """
        q_ids = self.encode(question)
        t_ids = self.encode(str(target_int))
        ids = [self.bos_id] + q_ids + [self.sep_id] + t_ids + [self.eos_id]
        sep_pos = 1 + len(q_ids)
        return ids, sep_pos

    def assert_corpus_covered(self, rows: Iterable[dict],
                              label: str = "corpus") -> None:
        """Hard-fail if any normalized char in `rows` is not in vocab.

        Run at trainer startup before any batch is built. The receipt
        says hard-fail; do not allow silent drops.
        """
        missing: set[str] = set()
        for r in rows:
            for c in normalize_text(r["question"]):
                if c not in self.char_to_id:
                    missing.add(c)
            for c in normalize_text(str(r["expected"])):
                if c not in self.char_to_id:
                    missing.add(c)
        if missing:
            repr_missing = [repr(c)[1:-1] for c in sorted(missing)]
            raise ValueError(
                f"{label} has {len(missing)} chars OOV vs declared vocab "
                f"({NORMALIZER_VERSION}): {repr_missing}. "
                f"Extend NORMALIZER_VERSION or _RESERVED_EXTRAS before train."
            )
