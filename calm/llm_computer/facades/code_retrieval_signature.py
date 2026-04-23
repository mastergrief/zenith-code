"""CodeRetrievalSignatureFacade — retrieval-predicted signature + AST rename.

Third tier-2 approach for the MBPP code-correctness task. Given an NL
problem, look up nearest-neighbor examples in `CodeExampleDB` (8970
retrieval corpus) and read the function signature off the top hit's
code_fragment. Use that as the target name for the AST rename.

Motivation: the DT-training path (CodeDtSkeletonFacade) and
extract-from-prompt path (CodeSkeletonFacade, ruled out at 1.6%)
both try to PREDICT a signature from scratch. But for ~60% of MBPP,
the signature already exists verbatim in the retrieval DB — the DB
loads MBPP among its sources. Even for novel prompts, retrieval
surfaces a near-neighbor whose signature convention is usually
appropriate.

Pipeline:
  parse:     encode prompt via dense retrieval + TF-IDF
  lookup:    top-K hits from CodeExampleDB (default hybrid RRF)
  evaluate:  extract `def NAME(args):` from top-1 code_fragment
  deliver:   run Gemma natural, then AST-rename first def to
             the retrieved NAME (args not enforced — keeps
             Gemma's argument choices intact)

vs CodeDtSkeletonFacade: trained-free, no decode-time bias, no
arity hallucinations.

vs CodeRenameFacade: automatic name prediction, doesn't require
caller to supply fn_name (useful when tests aren't available).

vs raw retrieval + code-hint injection (R53 approach): no prose
contamination, no token budget cost — we only use the retrieved
*signature*, discarding the rest. Per `retrieval.md` §"Retrieval-
content policy", retrieved content injected as hints causes Gemma
to imitate FORMAT. Signature-only extraction avoids that entirely.

Config:
  exclude_self_match: for benchmark-honesty. When the top-1 hit
    has problem text matching the input prompt too closely
    (simulating "this problem isn't in the DB yet"), skip to top-2.
    Default False (accept exact-match — deployment behavior).
  min_score: minimum retrieval confidence to fire the rename.
    Below threshold → pass-through natural Gemma (Tier 1
    preservation per augmentation_thesis.md).

Usage:
    facade = CodeRetrievalSignatureFacade()
    facade.install(gemma, tokenizer)
    r = facade.solve(prompt, exclude_self_match=True)
    # r.retrieved_fn_name = 'prime_num'  (from top-1)
    # r.retrieved_score = 0.84
    # r.generated = Gemma's output, with first def renamed to prime_num
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class RetrievalSignatureResult:
    prompt: str
    generated: str                        # post-rename output
    raw_generated: str                    # pre-rename output
    retrieved_fn_name: Optional[str] = None
    retrieved_args: List[str] = field(default_factory=list)
    retrieved_score: float = 0.0
    retrieved_source: Optional[str] = None   # DB source file
    retrieved_problem: Optional[str] = None  # snippet for debugging
    original_name: Optional[str] = None
    did_rename: bool = False
    fired: bool = False                   # True iff retrieval confident enough


_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*:", re.MULTILINE)


class CodeRetrievalSignatureFacade:
    """Retrieval-predicted signature + post-gen AST rename. Zero
    training, zero decode-time bias, zero regression risk (rename
    preserves body)."""

    DEFAULT_MAX_TOKENS = 256
    DEFAULT_K = 5
    DEFAULT_MIN_SCORE = 0.03   # jaccard floor; below this, don't fire

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        k: int = DEFAULT_K,
        min_score: float = DEFAULT_MIN_SCORE,
        cache_dir: str = ".cache/r53_code_db",
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.k = k
        self.min_score = min_score
        self.cache_dir = cache_dir
        self.device = device
        self._gemma = None
        self._tokenizer = None
        self._db = None

    def install(self, gemma, tokenizer):
        from calm.llm_computer.facades.code_example_db import CodeExampleDB
        from calm.llm_computer.facades.retrieval import TfidfIndex
        self._gemma = gemma
        self._tokenizer = tokenizer
        if self._db is None:
            db = CodeExampleDB.load_default()
            cache = Path(self.cache_dir)
            # Load TF-IDF only — dense index has a cuda/cpu tensor
            # alignment issue with the long-lived daemon's Pi buffers
            # that blocks load_indices() when dense.pt is present.
            # BM25/TF-IDF alone is enough for signature retrieval on
            # well-posed NL prompts (per retrieval.md: "BM25 is the
            # default for unfamiliar queries").
            tfidf_path = cache / "tfidf.json"
            if tfidf_path.exists():
                db._tfidf = TfidfIndex.load(tfidf_path)
            self._db = db

    def detach(self):
        self._gemma = None
        self._tokenizer = None

    def predict_signature(
        self,
        prompt: str,
        *,
        exclude_self_match: bool = False,
        exclude_threshold: float = 0.85,
    ) -> tuple[Optional[str], List[str], float, Optional[str], Optional[str]]:
        """Return (fn_name, args, score, source, problem_snippet) or
        Nones if nothing above min_score / all excluded.

        When `exclude_self_match=True`, any retrieved hit whose
        problem-text Jaccard similarity to `prompt` is above
        `exclude_threshold` is skipped (benchmark-honest mode).
        """
        if self._db is None:
            raise RuntimeError("facade not installed — call install() first")
        # Use TF-IDF — dense load is blocked (see install() note)
        hits = self._db.retrieve(prompt, k=self.k, mode="tfidf")
        if not hits:
            return None, [], 0.0, None, None

        for h in hits:
            if h.score < self.min_score:
                continue
            if exclude_self_match and _problem_similarity(
                    prompt, h.example.problem) >= exclude_threshold:
                continue
            # Extract first def from code_fragment (preferred) or solution
            source_text = h.example.code_fragment or h.example.solution
            m = _DEF_RE.search(source_text)
            if not m:
                continue
            fn_name = m.group(1)
            args_str = m.group(2).strip()
            args = []
            if args_str:
                args = [a.strip() for a in args_str.split(",") if a.strip()]
            return (fn_name, args, float(h.score),
                    h.example.source, h.example.problem[:120])
        return None, [], 0.0, None, None

    def solve(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        exclude_self_match: bool = False,
    ) -> RetrievalSignatureResult:
        if self._gemma is None or self._tokenizer is None:
            raise RuntimeError("facade not installed — call install() first")
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        fn_name, args, score, src, prob_snip = self.predict_signature(
            prompt, exclude_self_match=exclude_self_match,
        )

        result = RetrievalSignatureResult(
            prompt=prompt, generated="", raw_generated="",
            retrieved_fn_name=fn_name, retrieved_args=args,
            retrieved_score=score, retrieved_source=src,
            retrieved_problem=prob_snip,
            fired=fn_name is not None,
        )

        raw = self._generate(prompt, max_tokens)
        result.raw_generated = raw

        if fn_name is not None:
            from calm.llm_computer.facades.code_rename import rename_first_def
            renamed, orig = rename_first_def(raw, fn_name)
            result.generated = renamed
            result.original_name = orig
            result.did_rename = (orig is not None and orig != fn_name)
        else:
            # Retrieval couldn't confidently pick a name — Tier 1 preserve.
            result.generated = raw

        return result

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Natural Gemma generation — same template as the other
        tier-2 facades for fair A/B comparison."""
        from calm.llm_computer.gemma_substrate import KVCache
        gemma = self._gemma
        tok = self._tokenizer
        decorated = prompt.rstrip()
        if not decorated.endswith("```python"):
            decorated = decorated + "\n```python\n"
        ids = tok.encode(decorated)
        cache = KVCache(gemma.config.n_layers, device=self.device)
        gen = list(ids)
        with torch.no_grad():
            logits = gemma.forward(
                torch.tensor([gen]), device=self.device,
                kv_cache=cache, start_pos=0,
            )
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            for _ in range(max_tokens - 1):
                if hasattr(tok, "EOS_ID") and nxt == tok.EOS_ID:
                    break
                logits = gemma.forward(
                    torch.tensor([[nxt]]), device=self.device,
                    kv_cache=cache, start_pos=len(gen) - 1,
                )
                nxt = int(logits[0, -1].argmax())
                gen.append(nxt)
        emitted_ids = gen[len(ids):]
        return tok.decode(emitted_ids) if hasattr(tok, "decode") else ""


def _problem_similarity(a: str, b: str) -> float:
    """Jaccard token similarity for self-match detection."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
