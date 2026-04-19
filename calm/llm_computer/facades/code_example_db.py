"""CodeExampleDB — the heavy retrieval DB for R53.

Loads verified (problem, solution) pairs from JSONL corpora and exposes
a retrieval interface for R53's CodeVerifierFacade. Consolidates:

  - agents/distill/data/coding_reasoning_claude.jsonl (547 hand-written)
  - agents/distill/data/claude_reasoning.jsonl (910 merged)
  - (extensible) any additional jsonl with messages-schema

Storage is in-memory for MVP. Retrieval uses tokenized bag-of-words
Jaccard similarity on problem statements. Upgradable to BPE-hash or
dense embeddings without changing the public API.

Each entry is keyed by a stable integer derived from the problem hash
so it can also flow into KnowledgeStore (persistent_knowledge.py) as a
(query_key, correct_value_index) pair when the solution index is
compiled into substrate weights via ReGLU step functions.

Usage:
    db = CodeExampleDB.load_default()
    hits = db.retrieve("write a function that parses CSV", k=3)
    for h in hits:
        print(h.problem, h.solution_preview)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]

# Loaded in priority order — first occurrence of a problem wins dedup.
# Hand-written Claude-authored corpora come first (highest quality);
# then curated HF datasets with per-source quality filters; then
# programmatic benchmarks; then 9B-authored language files; finally
# the broad HF prefilter for long-tail paraphrase coverage.
DEFAULT_CORPORA = [
    # Hand-written / high-quality Claude
    REPO_ROOT / "agents/distill/data/coding_reasoning_claude.jsonl",       # 547 hand-written
    REPO_ROOT / "agents/distill/data/claude_reasoning.jsonl",              # 910 merged+filtered
    # Curated code benchmarks with test cases
    REPO_ROOT / "agents/distill/data/mbpp.jsonl",                          # 974 MBPP (has tests)
    REPO_ROOT / "agents/distill/data/humanevalplus.jsonl",                 # 164 HumanEval+
    REPO_ROOT / "agents/distill/data/bigcodebench.jsonl",                  # 1140 BigCodeBench (tests)
    REPO_ROOT / "agents/distill/data/codecontests.jsonl",                  # ~2-4K CodeContests Python3
    # Filtered HF Opus-reasoning (code-only + quality gate)
    REPO_ROOT / "agents/distill/data/nohurry_code.jsonl",                  # 106 code-category only
    REPO_ROOT / "agents/distill/data/crownelius.jsonl",                    # 265 quality-filtered
    REPO_ROOT / "agents/distill/data/claude_reasoning_hf_raw.jsonl",       # 886 TeichAI
    # Synthetic generators (this project)
    REPO_ROOT / "agents/distill/data/multi_step_code.jsonl",               # initial 48
    REPO_ROOT / "agents/distill/data/generated/algorithms.jsonl",          # framework output
    REPO_ROOT / "agents/distill/data/generated/stdlib.jsonl",
    REPO_ROOT / "agents/distill/data/generated/bug_fix.jsonl",
    REPO_ROOT / "agents/distill/data/generated/security.jsonl",
    REPO_ROOT / "agents/distill/data/generated/param_math.jsonl",          # heavy param sweeps
    REPO_ROOT / "agents/distill/data/generated/regex.jsonl",
    REPO_ROOT / "agents/distill/data/generated/data_structures.jsonl",
    REPO_ROOT / "agents/distill/data/generated/datetime_utils.jsonl",      # date/time correctness
    REPO_ROOT / "agents/distill/data/generated/functional.jsonl",          # HOF idioms
    # 9B-generated per-language
    REPO_ROOT / "agents/distill/data/python.jsonl",                        # 25 9B
    REPO_ROOT / "agents/distill/data/typescript.jsonl",                    # 39 9B
    REPO_ROOT / "agents/distill/data/rust.jsonl",                          # 53 9B
    # Broad long-tail (raw HF prefilter — lowest priority after dedup)
    REPO_ROOT / "agents/distill/data/claude_reasoning_prefilter.jsonl",    # 2940 raw HF
]

# Tokens that carry signal for retrieval. Everything else (stop words,
# filler) is dropped before Jaccard. Cheap heuristic — upgrade to TF-IDF
# or embeddings if retrieval precision plateaus.
_STOPWORDS = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for",
    "from", "get", "got", "had", "has", "have", "having", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "like", "may",
    "me", "might", "must", "my", "need", "needs", "no", "not", "of",
    "on", "or", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "up", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "would", "you", "your",
})


@dataclass(frozen=True)
class CodeExample:
    """One retrieval result."""
    key: int                    # stable integer hash of the problem
    problem: str
    solution: str
    source: str                 # path of the jsonl it came from
    tokens: frozenset[str]      # precomputed for Jaccard

    @property
    def solution_preview(self) -> str:
        """Code-only view of the solution for injection as a retrieval
        hint. Strips <think> blocks, "Verified test cases" trailers,
        and prose — returns ONLY the code so Gemma imitates the code,
        not the explanation style.

        Fix per R53.2b eval: raw solution_preview contained <think>
        blocks + prose headers. Prepending these as "related solution"
        hints caused Gemma to imitate the thinking-style format and
        emit prose instead of code, making its output unparseable.
        """
        s = self.solution
        # Drop <think>...</think> blocks (may be multiple)
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        # Drop trailing "Verified test cases" / "Sample I/O" sections
        # (from our generators and competitive-programming corpora)
        for trailer_pat in (r"\*\*Verified test cases",
                             r"\*\*Sample I/O",
                             r"\*\*Unit tests",
                             r"\*\*Test harness"):
            s = re.split(trailer_pat, s, maxsplit=1)[0]
        # Prefer the first ```python-fenced block — pure code
        m = re.search(r"```(?:python)?\s*\n(.*?)```", s, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: slice from first def/class
        m = re.search(r"(^|\n)(def |class |import |from \w+ import )", s)
        if m:
            return s[m.start():].strip()
        # Last resort: raw (trimmed) text
        return s.strip()


@dataclass
class RetrievalHit:
    """An example ranked against a query."""
    example: CodeExample
    score: float                # Jaccard in [0, 1]


@dataclass
class CodeExampleDB:
    """In-memory code-example retrieval DB.

    Deduplicates on problem hash (`CodeExample.key`) across corpora —
    first occurrence wins. Load higher-quality corpora first so their
    solutions survive when the same problem appears in multiple files.

    Retrieval modes (selected automatically at query time):

      - Jaccard (default, always available): fast, no indexing, weak
        on paraphrase and rare-term weighting.
      - TF-IDF sparse (via `build_tfidf()`): rare-term weighted cosine
        over posting lists; strong on exact-token match including
        rare technical terms.
      - Dense Gemma-encoded (via `build_dense(m, tok)`): semantic
        paraphrase via mean-pooled hidden states; strong on synonyms
        and rephrasing.
      - Hybrid RRF (auto when both built): reciprocal-rank fusion of
        TF-IDF top-k + Dense top-k for consensus-promoted results.

    Construction order: load JSONL → dedup → `build_tfidf()` (CPU)
    → `build_dense(m, tok)` (GPU, needs daemon). Persist each via
    `save_indices()` / `load_indices()` to skip rebuild on next run.
    """
    examples: List[CodeExample] = field(default_factory=list)
    _seen_keys: set[int] = field(default_factory=set)
    # Optional retrieval indices. Populated by build_* methods.
    _tfidf: Optional["TfidfIndex"] = None      # type: ignore[name-defined]
    _dense: Optional["DenseIndex"] = None      # type: ignore[name-defined]

    # ----- construction -----

    @classmethod
    def load_default(cls) -> "CodeExampleDB":
        return cls.load_paths(DEFAULT_CORPORA)

    @classmethod
    def load_paths(cls, paths: Iterable[Path]) -> "CodeExampleDB":
        db = cls()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            db.ingest_jsonl(p)
        return db

    def ingest_jsonl(self, path: Path) -> int:
        """Load a JSONL with {messages: [system, user, assistant, ...]}
        schema. Returns number of unique examples added after dedup."""
        added = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages", [])
                user = next(
                    (m["content"] for m in msgs if m.get("role") == "user"),
                    None,
                )
                asst = next(
                    (m["content"] for m in msgs if m.get("role") == "assistant"),
                    None,
                )
                if not user or not asst:
                    continue
                ex = _make_example(user, asst, str(path))
                if ex.key in self._seen_keys:
                    continue
                self._seen_keys.add(ex.key)
                self.examples.append(ex)
                added += 1
        return added

    # ----- retrieval -----

    def retrieve(self, query: str, k: int = 3,
                 min_score: float = 0.05,
                 mode: str = "auto",
                 sparse_k: int = 10,
                 dense_k: int = 10,
                 dense_m=None, dense_tok=None) -> List[RetrievalHit]:
        """Top-k examples most similar to `query`.

        Modes:
          - "jaccard": legacy token-overlap (default fallback).
          - "tfidf":   sparse TF-IDF cosine (requires build_tfidf()).
          - "dense":   dense Gemma-encoded cosine (requires build_dense()).
          - "hybrid":  RRF(tfidf, dense) — needs both built.
          - "auto":    hybrid if both built, else whichever is available,
                       else jaccard.

        For "dense" / "hybrid" modes, pass `dense_m` and `dense_tok`
        (the loaded Gemma + tokenizer) so the query can be encoded.
        """
        actual_mode = self._resolve_mode(mode, dense_m is not None)
        if actual_mode == "jaccard":
            return self._retrieve_jaccard(query, k, min_score)
        if actual_mode == "tfidf":
            return self._retrieve_from_ranked(
                self._tfidf.query(query, k=k), k)
        if actual_mode == "dense":
            return self._retrieve_from_ranked(
                self._dense.query(query, dense_m, dense_tok, k=k), k)
        if actual_mode == "hybrid":
            from calm.llm_computer.facades.retrieval import rrf_fuse
            tfidf_ranked = self._tfidf.query(query, k=sparse_k)
            dense_ranked = self._dense.query(
                query, dense_m, dense_tok, k=dense_k)
            fused = rrf_fuse([tfidf_ranked, dense_ranked], top_k=k)
            return self._retrieve_from_ranked(fused, k)
        return []

    def _resolve_mode(self, requested: str,
                      have_dense_engine: bool) -> str:
        """Pick the actual retrieval mode based on availability."""
        have_tfidf = self._tfidf is not None
        have_dense = self._dense is not None and have_dense_engine
        if requested != "auto":
            # Validate requested mode is available; fall back if not.
            if requested == "tfidf" and not have_tfidf:
                return "jaccard"
            if requested == "dense" and not have_dense:
                return "jaccard"
            if requested == "hybrid" and not (have_tfidf and have_dense):
                if have_tfidf:
                    return "tfidf"
                if have_dense:
                    return "dense"
                return "jaccard"
            return requested
        # Auto
        if have_tfidf and have_dense:
            return "hybrid"
        if have_tfidf:
            return "tfidf"
        if have_dense:
            return "dense"
        return "jaccard"

    def _retrieve_jaccard(self, query: str, k: int,
                          min_score: float) -> List[RetrievalHit]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: List[RetrievalHit] = []
        for ex in self.examples:
            denom = len(q_tokens | ex.tokens)
            if denom == 0:
                continue
            score = len(q_tokens & ex.tokens) / denom
            if score >= min_score:
                scored.append(RetrievalHit(example=ex, score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def _retrieve_from_ranked(
        self, ranked: List[tuple], k: int,
    ) -> List[RetrievalHit]:
        """Convert a [(doc_idx, score), ...] list to RetrievalHit objects."""
        out: List[RetrievalHit] = []
        for doc_idx, score in ranked[:k]:
            if 0 <= doc_idx < len(self.examples):
                out.append(RetrievalHit(
                    example=self.examples[doc_idx], score=float(score)))
        return out

    # -------- retrieval index builders --------

    def build_tfidf(self, include_solution: bool = True,
                    problem_weight: int = 3) -> None:
        """Build the sparse index from current `examples` list.

        include_solution=True indexes problem + solution so queries can
        match by symptom or by function signature. problem_weight
        controls how many times the problem text is repeated relative
        to the solution (default 3×) — higher = more weight on
        user-facing statement.

        Run once after ingest; O(total tokens) time.
        """
        from calm.llm_computer.facades.retrieval import TfidfIndex
        idx = TfidfIndex()
        def _docs():
            for ex in self.examples:
                if include_solution:
                    yield (
                        (ex.problem + " ") * problem_weight
                        + ex.solution
                    )
                else:
                    yield ex.problem
        idx.build(_docs())
        self._tfidf = idx

    def build_dense(self, m, tok, max_len: int = 256,
                    batch_size: int = 8) -> None:
        """Build dense Gemma-encoded index. Requires loaded substrate
        + tokenizer. Time: ~5-30 min for 10K examples depending on
        prefill speed + batch."""
        from calm.llm_computer.facades.retrieval import DenseIndex
        idx = DenseIndex(d_model=m.config.d_model)
        idx.build(self.examples, m, tok, max_len=max_len,
                  batch_size=batch_size)
        self._dense = idx

    def save_indices(self, dirpath: Path,
                      dense_quantize: bool = True) -> None:
        """Persist built indices under a directory. TF-IDF as JSON,
        dense vectors as fp16 .pt. When dense_quantize=True (default),
        also writes a tq4 sibling for ~4× smaller storage; at load time
        the tq4 dequant adds a one-off ~300 ms."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        if self._tfidf is not None:
            self._tfidf.save(dirpath / "tfidf.json")
        if self._dense is not None:
            self._dense.save(dirpath / "dense.pt",
                              quantize=dense_quantize)

    def load_indices(self, dirpath: Path) -> None:
        """Reload previously-saved indices. Does not rebuild on mismatch
        — caller's responsibility to keep DB and indices in sync."""
        from calm.llm_computer.facades.retrieval import (
            DenseIndex, TfidfIndex,
        )
        dirpath = Path(dirpath)
        tfidf_path = dirpath / "tfidf.json"
        dense_path = dirpath / "dense.pt"
        if tfidf_path.exists():
            self._tfidf = TfidfIndex.load(tfidf_path)
        if dense_path.exists():
            self._dense = DenseIndex.load(dense_path)

    def has_tfidf(self) -> bool:
        return self._tfidf is not None

    def has_dense(self) -> bool:
        return self._dense is not None

    # ----- DB stats -----

    def __len__(self) -> int:
        return len(self.examples)

    def summary(self) -> dict:
        sources: dict[str, int] = {}
        for ex in self.examples:
            sources[ex.source] = sources.get(ex.source, 0) + 1
        return {
            "total": len(self.examples),
            "sources": sources,
            "avg_problem_len": (
                sum(len(e.problem) for e in self.examples) / len(self.examples)
                if self.examples else 0
            ),
            "avg_solution_len": (
                sum(len(e.solution) for e in self.examples) / len(self.examples)
                if self.examples else 0
            ),
        }


# ---- module-private helpers ----

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    """Lowercased alphanumeric tokens minus stopwords, length >= 3."""
    toks = _TOKEN_RE.findall(text.lower())
    return frozenset(t for t in toks if len(t) >= 3 and t not in _STOPWORDS)


def _stable_key(problem: str) -> int:
    """Stable non-negative int key from problem hash. Fits in 63 bits."""
    h = hashlib.blake2b(problem.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def _make_example(problem: str, solution: str, source: str) -> CodeExample:
    return CodeExample(
        key=_stable_key(problem),
        problem=problem,
        solution=solution,
        source=source,
        tokens=_tokenize(problem),
    )
