"""CodeDtSkeletonFacade — trained-DT signature prediction + Gemma decode bias.

End-to-end path for the v14 code-skeleton DT checkpoint (0.20 greedy /
0.31 beam honest val at 563 held-out). Not the regex-parser
CodeSkeletonFacade (ruled out at 1.6% — `facades/code_skeleton.py`).

Pipeline:
  parse        → predict `def FN(<args>):` via DT greedy decode
  evaluate     → replace `FN` with caller-supplied fn_name; build
                 `def <fn_name>(<args>):` skeleton string
  deliver      → multi-token step-through BPE bias at Gemma decode
                 (same mechanism as Icd10RecallFacade for text answers,
                 R46.2/R22c for integer answers)

The caller provides `fn_name` (MBPP tests PIN the function name via
`assert <name>(...)` so it's known a priori). DT predicts the arg
list; the facade stitches the skeleton and biases Gemma to emit it
verbatim before continuing to generate the body.

Automatic tier-1 preservation: on prompts where DT's predicted
skeleton doesn't parse as `def FN(...):`, facade declines and Gemma
generates unbiased.

Usage:

    facade = CodeDtSkeletonFacade(
        checkpoint_path="calm/hrm/checkpoints/dt_code_skel_best.pt",
    )
    facade.install(gemma, tokenizer)

    result = facade.solve(
        "Write a function to count vowels in a string.",
        fn_name="count_vowels",
    )
    # result.predicted_args = ["text"]  (example, DT-dependent)
    # result.skeleton = "def count_vowels(text):"
    # result.generated = "def count_vowels(text):\n    return sum(..."
    # result.used_bias = True
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class CodeDtSkeletonResult:
    prompt: str
    fn_name: str
    predicted_args: List[str] = field(default_factory=list)
    predicted_skeleton: Optional[str] = None  # raw DT output, e.g. "def FN(x):"
    skeleton: Optional[str] = None            # with fn_name substituted
    generated: str = ""
    used_bias: bool = False


_SKEL_RE = re.compile(r"def\s+FN\s*\(([^)]*)\)\s*:")


class CodeDtSkeletonFacade:
    """NL → signature via trained DT, then Gemma decode bias."""

    DEFAULT_BOOST = 50.0
    DEFAULT_MAX_TOKENS = 256  # enough for signature + function body
    DT_MAX_GEN = 40           # char budget for skeleton decode

    def __init__(
        self,
        checkpoint_path: str | Path = "calm/hrm/checkpoints/dt_code_skel_best.pt",
        boost: float = DEFAULT_BOOST,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str = "cuda",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.boost = boost
        self.max_tokens = max_tokens
        self.device = device
        self._gemma = None
        self._tokenizer = None
        self._dt = None

    # --- lifecycle ---

    def install(self, gemma, tokenizer):
        from calm.llm_computer.dt_install import load_dt_checkpoint
        self._gemma = gemma
        self._tokenizer = tokenizer
        if self._dt is None:
            dt_model, ckpt = load_dt_checkpoint(
                self.checkpoint_path, device=self.device,
            )
            self._dt = dt_model
            self._ckpt_meta = {
                "val_autoreg": ckpt.get("val_autoreg"),
                "epoch": ckpt.get("epoch"),
            }

    def detach(self):
        self._gemma = None
        self._tokenizer = None
        # Keep DT loaded — cheap, reuse across install/detach cycles.

    # --- DT inference ---

    @torch.no_grad()
    def predict_skeleton(self, prompt: str) -> Optional[str]:
        """Run DT greedy decode on prompt; return raw `def FN(args):` or None."""
        from calm.hrm.code_dt_data import (
            code_tokenize, code_detokenize, _CODE_CHAR_TO_ID, _clean_prob,
        )
        if self._dt is None:
            raise RuntimeError("facade not installed — call install() first")

        # Match training-time preprocessing: clean + truncate to 180 chars.
        cleaned = _clean_prob(prompt, max_len=180)
        if cleaned is None:
            return None

        sep = _CODE_CHAR_TO_ID["<sep>"]
        eos = _CODE_CHAR_TO_ID["<eos>"]
        prefix = code_tokenize(cleaned, add_bos=True, add_eos=False) + [sep]

        pos_limit = self._dt.config.max_len
        if len(prefix) >= pos_limit - 5:
            # Truncate from the left, keeping BOS
            prefix = [prefix[0]] + prefix[-(pos_limit - 6):]

        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        gen_ids = self._dt.decode_greedy_cached(
            ids, max_gen=self.DT_MAX_GEN, eos_token=eos,
        )
        gen_list = gen_ids[0].tolist() if gen_ids.numel() > 0 else []
        text = code_detokenize(gen_list)
        return text or None

    @staticmethod
    def parse_skeleton(raw: str) -> Optional[List[str]]:
        """Parse `def FN(<args>):` → list of arg names. Returns None on miss."""
        m = _SKEL_RE.search(raw)
        if not m:
            return None
        args_str = m.group(1).strip()
        if not args_str:
            return []
        return [a.strip() for a in args_str.split(",") if a.strip()]

    # --- solve / deliver ---

    def solve(
        self,
        prompt: str,
        fn_name: str,
        *,
        max_tokens: Optional[int] = None,
        boost: Optional[float] = None,
        use_bias: bool = True,
    ) -> CodeDtSkeletonResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        boost = boost if boost is not None else self.boost

        result = CodeDtSkeletonResult(prompt=prompt, fn_name=fn_name)

        raw_skel = self.predict_skeleton(prompt)
        result.predicted_skeleton = raw_skel
        parsed = self.parse_skeleton(raw_skel) if raw_skel else None

        bias_ids: list[int] = []
        fire_bias = False
        if use_bias and parsed is not None:
            result.predicted_args = parsed
            result.skeleton = f"def {fn_name}({', '.join(parsed)}):"
            bias_ids = self._skeleton_to_gemma_tokens(result.skeleton)
            fire_bias = bool(bias_ids)

        result.used_bias = fire_bias
        result.generated = self._generate(
            prompt, bias_ids if fire_bias else [], boost, max_tokens,
        )
        return result

    def _skeleton_to_gemma_tokens(self, text: str) -> list[int]:
        """Gemma-BPE tokenize the skeleton. Strip BOS + leading ▁ so the
        first bias token is the first visible code char.

        Note: skeleton always starts with `def` which tokenizes as `▁def`
        (single BPE). We keep the ▁def token because Gemma's natural
        continuation on a code prompt often opens with a fence, not a
        bare `def` — bias needs to fire from the first token, including
        whatever leading-space/underscore Gemma's tokenizer assigns.
        """
        ids = self._tokenizer.encode(text)
        # Strip BOS only. Leading ▁def merges the leading space into the
        # word token, so we want to keep it (unlike integer-answer
        # facades which strip ▁ — see compute_facades.md §"▁-strip +
        # POST_BIAS_BUDGET").
        if ids and ids[0] == 2:
            ids = ids[1:]
        return ids

    def _generate(
        self,
        prompt: str,
        bias_token_ids: list[int],
        boost: float,
        max_tokens: int,
    ) -> str:
        """Gemma multi-token step-through bias. Same template as
        Icd10RecallFacade._generate, adapted for code skeleton tokens.

        Appends a '```python\n' marker before the decode so the first
        bias token aligns with Gemma's first emitted code token.
        Without the marker, Gemma emits fence + newline first and the
        bias lands on the wrong positions.
        """
        from calm.llm_computer.gemma_substrate import KVCache

        gemma = self._gemma
        tok = self._tokenizer
        # Format prompt to push Gemma toward immediate code emission.
        decorated = prompt.rstrip()
        if not decorated.endswith("```python"):
            decorated = decorated + "\n```python\n"

        ids = tok.encode(decorated)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        bias_idx = 0 if bias_token_ids else -1

        with torch.no_grad():
            logits = gemma.forward(
                torch.tensor([gen]), device=self.device,
                kv_cache=cache, start_pos=0,
            )
            if 0 <= bias_idx < len(bias_token_ids):
                logits[0, -1, bias_token_ids[bias_idx]] += boost
                bias_idx += 1
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)

            for _ in range(max_tokens - 1):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
                    break
                logits = gemma.forward(
                    torch.tensor([[nxt]]), device=self.device,
                    kv_cache=cache, start_pos=len(gen) - 1,
                )
                if 0 <= bias_idx < len(bias_token_ids):
                    logits[0, -1, bias_token_ids[bias_idx]] += boost
                    bias_idx += 1
                nxt = int(logits[0, -1].argmax())
                gen.append(nxt)

        emitted_ids = gen[len(ids):]
        return tok.decode(emitted_ids) if hasattr(tok, "decode") else ""
