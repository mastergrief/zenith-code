"""Hybrid retrieval for CodeExampleDB — TF-IDF + dense + RRF.

Three components, plugged into `CodeExampleDB.retrieve()`:

  1. TfidfIndex   — hand-rolled sparse retrieval (no sklearn dep).
                    Rewards rare-term matches (SSRF, Levenshtein, ...).
  2. DenseIndex   — Gemma-encoded dense vectors. Rewards semantic
                    paraphrase ("SSRF" vs "server-side request forgery").
                    Substrate-native: same representation the L24 card
                    will read at install time (R53.6).
  3. rrf_fuse     — Reciprocal Rank Fusion across ranked lists. Merges
                    sparse + dense results with consensus promotion.

Each index is optional — `CodeExampleDB` falls back to Jaccard if
neither is built. Build order: TF-IDF first (CPU, <5s), Dense second
(GPU, ~1-5 min for 10K examples), then fuse at query time.

Design for 10K+ examples:
  - TF-IDF: in-memory sparse dict-of-dicts. ~30K vocab, ~100 tokens/doc.
    Memory: ~80 MB for 10K docs. Query: O(|q| × |postings|) ≈ 10 ms.
  - Dense: (10K, d_model) float16 = 40 MB for Gemma 4 E4B (d=2560).
    Query: single matmul (q @ docs.T), ~5 ms on GPU, ~50 ms on CPU.
  - Cache both to disk (.pt for dense, .json for TF-IDF) so reloading
    an already-indexed DB is <1s.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# Lazy import — CodeExample lives in a sibling module. Using TYPE_CHECKING
# guard avoids circular-import headaches at module load time.
from calm.llm_computer.facades.code_example_db import CodeExample


# -------------------------------------------------------------
# TF-IDF (sparse, hand-rolled, no sklearn)
# -------------------------------------------------------------

_TFIDF_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

_TFIDF_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "with", "you", "your", "was", "were", "will",
})


def _tfidf_tokenize(text: str) -> List[str]:
    """Case-fold + alphanumeric tokens length >= 2 + drop stopwords."""
    return [
        t for t in (m.group(0).lower()
                    for m in _TFIDF_TOKEN_RE.finditer(text))
        if len(t) >= 2 and t not in _TFIDF_STOPWORDS
    ]


@dataclass
class TfidfIndex:
    """Sparse retrieval index supporting TF-IDF cosine AND BM25 scoring.

    Two views of the corpus (all precomputed at build time):
      - `_term_to_postings[term]` -> {doc_idx: (tf, tfidf_weight)}
        Stores raw tf (for BM25) + precomputed tfidf weight (for cosine).
      - `_doc_norms[doc_idx]`     -> precomputed L2 norm of tfidf vector
      - `_doc_lens[doc_idx]`      -> token count (for BM25 length-norm)
      - `_idf[term]`              -> IDF score

    At query time pick the scorer:
      - `.query(text, k, scorer="tfidf")` — classic cosine
      - `.query(text, k, scorer="bm25")`  — BM25 (log-saturation on tf + length-norm)

    BM25 is typically more robust for search queries; TF-IDF is lighter
    and closer to what downstream embeddings expect. Default: bm25.
    """
    _term_to_postings: Dict[str, Dict[int, Tuple[int, float]]] = field(default_factory=dict)
    _doc_norms: List[float] = field(default_factory=list)
    _doc_lens: List[int] = field(default_factory=list)
    _idf: Dict[str, float] = field(default_factory=dict)
    _n_docs: int = 0
    _avgdl: float = 0.0

    # BM25 hyperparameters (standard defaults)
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    def build(self, docs: Iterable[str]) -> None:
        """Build the index over a corpus. Each doc is a raw string; the
        document index is its position in `docs`."""
        postings_raw: Dict[str, Dict[int, int]] = {}
        doc_lens: List[int] = []

        for doc_idx, text in enumerate(docs):
            tokens = _tfidf_tokenize(text)
            doc_lens.append(len(tokens))
            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1
            for t, c in term_counts.items():
                postings_raw.setdefault(t, {})[doc_idx] = c

        n_docs = len(doc_lens)
        self._n_docs = n_docs
        self._doc_lens = doc_lens
        self._avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0

        # IDF: smoothed variant (sklearn-style) — works for both scorers.
        idf: Dict[str, float] = {}
        for t, p in postings_raw.items():
            df = len(p)
            idf[t] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # Store (raw_tf, tfidf_weight) per posting — BM25 uses raw tf,
        # cosine uses precomputed tfidf. Space cost is modest; index
        # fits in memory at 10K+ docs without issue.
        weighted: Dict[str, Dict[int, Tuple[int, float]]] = {}
        for t, p in postings_raw.items():
            idf_t = idf[t]
            inner: Dict[int, Tuple[int, float]] = {}
            for d, tf in p.items():
                w_tfidf = (1.0 + math.log(tf)) * idf_t
                inner[d] = (tf, w_tfidf)
            weighted[t] = inner

        # Precompute doc L2 norms for cosine
        doc_sq_sums = [0.0] * n_docs
        for t, inner in weighted.items():
            for d, (tf, w) in inner.items():
                doc_sq_sums[d] += w * w
        doc_norms = [math.sqrt(s) if s > 0 else 1.0 for s in doc_sq_sums]

        self._term_to_postings = weighted
        self._idf = idf
        self._doc_norms = doc_norms

    def query(self, text: str, k: int = 10,
              scorer: str = "bm25") -> List[Tuple[int, float]]:
        """Return top-k (doc_idx, score) for the query string.

        scorer="tfidf": classic cosine over tfidf-weighted vectors.
                        Normalized to [0, 1]. Interpretable.
        scorer="bm25":  Okapi BM25 with log-saturated tf and length
                        normalization. Typically more robust for
                        keyword search. Unbounded score. Default.
        """
        if self._n_docs == 0:
            return []
        tokens = _tfidf_tokenize(text)
        if not tokens:
            return []
        q_counts: Dict[str, int] = {}
        for t in tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        if scorer == "bm25":
            return self._query_bm25(q_counts, k)
        if scorer == "tfidf":
            return self._query_tfidf(q_counts, k)
        raise ValueError(f"unknown scorer: {scorer!r}")

    def _query_tfidf(self, q_counts: Dict[str, int], k: int
                     ) -> List[Tuple[int, float]]:
        """Classic TF-IDF cosine similarity."""
        q_weights: Dict[str, float] = {}
        for t, tf in q_counts.items():
            idf_t = self._idf.get(t)
            if idf_t is None:
                continue
            q_weights[t] = (1.0 + math.log(tf)) * idf_t
        if not q_weights:
            return []
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0:
            return []

        scores: Dict[int, float] = {}
        for t, q_w in q_weights.items():
            posting = self._term_to_postings.get(t)
            if posting is None:
                continue
            for d, (tf, d_w) in posting.items():
                scores[d] = scores.get(d, 0.0) + q_w * d_w

        items: List[Tuple[int, float]] = []
        for d, num in scores.items():
            denom = q_norm * self._doc_norms[d]
            if denom == 0:
                continue
            items.append((d, num / denom))

        items.sort(key=lambda pair: pair[1], reverse=True)
        return items[:k]

    def _query_bm25(self, q_counts: Dict[str, int], k: int
                    ) -> List[Tuple[int, float]]:
        """Okapi BM25. Per standard formula:
            score(d, q) = sum over t in q:
                idf(t) * tf(t,d) * (k1 + 1)
                         / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
        """
        k1, b = self.BM25_K1, self.BM25_B
        scores: Dict[int, float] = {}
        avgdl = self._avgdl if self._avgdl > 0 else 1.0
        for term, q_tf in q_counts.items():
            idf_t = self._idf.get(term)
            posting = self._term_to_postings.get(term)
            if idf_t is None or posting is None:
                continue
            for d, (tf, _) in posting.items():
                denom_norm = 1.0 - b + b * (self._doc_lens[d] / avgdl)
                contrib = idf_t * (tf * (k1 + 1.0)) / (tf + k1 * denom_norm)
                scores[d] = scores.get(d, 0.0) + contrib

        items = list(scores.items())
        items.sort(key=lambda pair: pair[1], reverse=True)
        return items[:k]

    # -------- persistence --------

    def save(self, path: Path) -> None:
        """Serialize to JSON. ~30-80 MB for 10K docs (depends on vocab).
        Format v2: postings entries are [doc, raw_tf, tfidf_w] triples
        (v1 used [doc, tfidf_w] pairs; loader accepts both)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "n_docs": self._n_docs,
            "avgdl": self._avgdl,
            "idf": self._idf,
            "doc_lens": self._doc_lens,
            "doc_norms": self._doc_norms,
            "postings": {
                t: [[d, tf, w] for d, (tf, w) in inner.items()]
                for t, inner in self._term_to_postings.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfIndex":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls()
        idx._n_docs = int(data["n_docs"])
        idx._avgdl = float(data.get("avgdl", 0.0))
        idx._idf = {k: float(v) for k, v in data["idf"].items()}
        idx._doc_norms = [float(x) for x in data["doc_norms"]]
        idx._doc_lens = [int(x) for x in data.get("doc_lens",
                         [0] * idx._n_docs)]

        postings: Dict[str, Dict[int, Tuple[int, float]]] = {}
        for t, lst in data["postings"].items():
            inner: Dict[int, Tuple[int, float]] = {}
            for entry in lst:
                if len(entry) == 3:                      # v2 format
                    d, tf, w = entry
                    inner[int(d)] = (int(tf), float(w))
                else:                                     # v1 format: [d, w]
                    d, w = entry
                    # tf unknown in v1; use 1 as safe BM25 fallback
                    inner[int(d)] = (1, float(w))
            postings[t] = inner
        idx._term_to_postings = postings
        return idx


# -------------------------------------------------------------
# Dense (Gemma-encoded)
# -------------------------------------------------------------

@dataclass
class DenseIndex:
    """Dense retrieval via mean-pooled Gemma hidden states.

    Stores one (d_model,) float16 vector per example, plus L2-normalized
    matrix for fast cosine via a single matmul.

    Build (one-time): tokenize each problem, forward through Gemma up to
    a chosen layer (default: middle layer, L21), mean-pool over tokens,
    L2-normalize. ~1-5 min for 10K examples depending on GPU.

    Query: embed the query with the same procedure, compute `q @ docs.T`,
    take top-k. On GPU at d=2560: <10 ms for 10K docs.
    """
    vectors: Optional[torch.Tensor] = None          # (N, d_model) float16, L2-normed
    layer_idx: int = 21
    d_model: int = 2560                              # Gemma 4 E4B
    device: str = "cuda"

    # GemmaTokenizer.encode is O(len × vocab_size) naive longest-prefix
    # scan over 262K tokens. _monkey_patch_fast_encode installs a trie-
    # backed O(len × max_token_len ≈ 20) replacement. Also truncate
    # input to _PRE_TOKENIZE_CHAR_LIMIT chars because pooling over 400
    # chars is plenty of retrieval signal and keeps per-text O(400).
    _PRE_TOKENIZE_CHAR_LIMIT: int = 400

    def _encode_texts(self, texts: List[str], m, tok,
                      max_len: int = 64,
                      batch_size: int = 8) -> torch.Tensor:
        """Encode a list of strings to L2-normalized vectors via mean-pooled
        Gemma token embeddings.

        Strategy: dedupe the union of token ids across all texts,
        dequant those ONCE into a CPU fp32 lookup table (sub-matrix
        of m.token_embd), then mean-pool per text on CPU.

        This avoids per-text GPU dequant round-trips (which have
        ~2-5 ms overhead each and would take 15+ min for 10K texts).
        With dedup + single batched dequant, 10K texts encode in
        ~5-10 seconds.

        Gemma's `token_embd` is Gemma's own input-embedding matrix,
        so the retrieval key space matches what a dense-retrieval
        card installed on prod Gemma (R53.6) would read from the
        residual at layer 0.
        """
        if not texts:
            return torch.empty(0, self.d_model, dtype=torch.float16)

        # --- Pass 0: install fast trie encoder on the tokenizer ---
        _monkey_patch_fast_encode(tok)

        # --- Pass 1: tokenize all, collect union of used ids ---
        # Truncate each text to _PRE_TOKENIZE_CHAR_LIMIT BEFORE calling
        # tok.encode — GemmaTokenizer's BPE is pathologically slow on
        # long code-fenced texts (seconds per call). First ~400 chars
        # carry the problem statement, which is the retrieval-relevant
        # signal. Solutions/tests in longer texts are already covered
        # by the TF-IDF side of the hybrid.
        text_ids: List[List[int]] = []
        all_used: set[int] = set()
        n_total = len(texts)
        limit = self._PRE_TOKENIZE_CHAR_LIMIT
        print(f"[dense] tokenizing {n_total} texts (char-limit={limit})...",
              flush=True)
        t0 = time.time()
        for i, text in enumerate(texts):
            truncated = text[:limit]
            ids = tok.encode(truncated)[:max_len]
            text_ids.append(ids)
            all_used.update(ids)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[dense]  tokenized {i + 1}/{n_total} "
                      f"({rate:.0f} texts/sec)", flush=True)

        if not all_used:
            return torch.zeros(
                len(texts), self.d_model, dtype=torch.float16)

        # --- Pass 2: single batched dequant of unique ids ---
        embd_matrix = m.token_embd
        dev = getattr(embd_matrix, "device", None) or self.device
        unique_ids = sorted(all_used)
        unique_ids_tensor = torch.tensor(
            unique_ids, dtype=torch.long, device=dev)

        # One GPU dequant covering every token we need. For Q6_K
        # embedding at vocab 262K × d 2560, a batch of 50K unique
        # tokens dequants in ~300 ms on RTX 4070.
        print(f"[dense] dequanting {len(unique_ids)} unique tokens...",
              flush=True)
        dequant = embd_matrix[unique_ids_tensor]      # (U, d_model) fp32
        # Move to CPU fp32 once; all per-text pooling happens CPU-side.
        lookup = dequant.float().cpu()
        id_to_row = {t: i for i, t in enumerate(unique_ids)}
        print(f"[dense] lookup table: {tuple(lookup.shape)} fp32 on CPU",
              flush=True)

        # --- Pass 3: per-text mean-pool from CPU lookup table ---
        all_vecs = torch.empty(
            len(texts), self.d_model, dtype=torch.float16)
        for i, ids in enumerate(text_ids):
            if not ids:
                all_vecs[i].zero_()
                continue
            rows = torch.tensor(
                [id_to_row[t] for t in ids], dtype=torch.long)
            emb = lookup.index_select(0, rows)        # (S, d_model)
            vec = emb.mean(dim=0)                      # (d_model,)
            vec = vec / (vec.norm() + 1e-8)
            all_vecs[i] = vec.half()
            if (i + 1) % 1000 == 0:
                print(f"[dense]  {i + 1}/{len(texts)} pooled", flush=True)

        return all_vecs

    def build(self, examples: List[CodeExample], m, tok,
              max_len: int = 256,
              batch_size: int = 8) -> None:
        """Build the dense index by encoding each example's problem text."""
        texts = [ex.problem for ex in examples]
        vectors = self._encode_texts(texts, m, tok,
                                     max_len=max_len,
                                     batch_size=batch_size)
        self.vectors = vectors  # (N, d_model) fp16 on cpu

    def query(self, text: str, m, tok, k: int = 10,
              max_len: int = 256) -> List[Tuple[int, float]]:
        """Encode query, compute cosine vs index, return top-k."""
        if self.vectors is None or self.vectors.numel() == 0:
            return []
        qv = self._encode_texts([text], m, tok,
                                max_len=max_len, batch_size=1)
        if qv.numel() == 0:
            return []
        q = qv[0]                                # (d_model,)
        D = self.vectors                         # (N, d_model)
        # Both are L2-normalized, so cosine = dot
        scores = (D.float() @ q.float()).tolist()
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:k]

    # -------- persistence --------

    def save(self, path: Path, quantize: bool = False) -> None:
        """Save the dense index to `path`.

        quantize=False (default): stores fp16 tensor. ~51 MB per 10K × 2560d.
        quantize=True:            also stores a tq4-quantized copy at
                                  `path.with_suffix('.tq4.pt')`. ~13 MB per 10K.
                                  At query time the tq4 can be dequant'd once
                                  at load via DenseIndex.load(..., prefer_tq4=True).
        """
        if self.vectors is None:
            raise RuntimeError("dense index empty — nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vectors": self.vectors,                        # fp16
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "format": "fp16",
        }, path)

        if quantize:
            from calm.llm_computer.tq4_torch import quantize_tq4
            tq4_path = path.with_suffix(".tq4.pt")
            # quantize_tq4 expects fp32 — upcast from fp16 briefly.
            vecs_fp32 = self.vectors.float().contiguous()
            # quantize_tq4 works on 1-D; flatten (N, d) → (N*d,)
            tq4 = quantize_tq4(vecs_fp32.view(-1))
            torch.save({
                "qs": tq4.qs,
                "d":  tq4.d,
                "shape": (self.vectors.shape[0], self.d_model),
                "layer_idx": self.layer_idx,
                "d_model": self.d_model,
                "format": "tq4",
            }, tq4_path)

    @classmethod
    def load(cls, path: Path, prefer_tq4: bool = False) -> "DenseIndex":
        """Load dense index. If prefer_tq4=True and a sibling .tq4.pt
        exists, dequantizes it into an fp16 tensor on load (one-time
        cost; query-time is same as fp16)."""
        path = Path(path)
        tq4_path = path.with_suffix(".tq4.pt")
        if prefer_tq4 and tq4_path.exists():
            from calm.llm_computer.tq4_torch import (
                Tq4Tensor, dequantize_tq4,
            )
            data = torch.load(tq4_path, map_location="cpu")
            shape = tuple(data["shape"])
            tq4 = Tq4Tensor(
                qs=data["qs"], d=data["d"],
                shape=(shape[0] * shape[1],),
            )
            # dequantize_tq4 auto-builds Pi + centroids when None passed.
            fp32 = dequantize_tq4(tq4)
            vectors = fp32.reshape(shape).half()
            idx = cls(
                layer_idx=int(data.get("layer_idx", 21)),
                d_model=int(data.get("d_model", 2560)),
            )
            idx.vectors = vectors
            return idx
        # Default: fp16 path
        data = torch.load(path, map_location="cpu")
        idx = cls(
            layer_idx=int(data.get("layer_idx", 21)),
            d_model=int(data.get("d_model", 2560)),
        )
        idx.vectors = data["vectors"]
        return idx


# -------------------------------------------------------------
# Fast tokenizer (trie-backed replacement for GemmaTokenizer.encode)
# -------------------------------------------------------------

def _monkey_patch_fast_encode(tok) -> None:
    """Replace GemmaTokenizer.encode with an O(len × max_token_len)
    trie-backed variant.

    The shipped GemmaTokenizer.encode does `for token, tid in sorted_tokens:
    if text[pos:pos+len(token)] == token: ...` — O(vocab × len) per call,
    which is ~100M ops per ~400-char text at vocab 262K. Our retrieval
    pipeline tokenizes 8970 texts → ~9e11 ops, multi-minute stalls.

    Fix: build a char-trie once (O(vocab × avg_token_len)). Each encode
    walks the trie from each pos remembering the longest match. O(len)
    per text regardless of vocab size.

    Idempotent: skips if already patched (checks `_fast_encode_trie`).
    """
    if hasattr(tok, "_fast_encode_trie"):
        return
    t0 = time.time()
    # Build trie from token_to_id
    root: Dict = {}
    for token, tid in tok.token_to_id.items():
        node = root
        for c in token:
            node = node.setdefault(c, {})
        # Use a sentinel key that can't appear as a char — "$$end"
        node["$$end"] = tid
    tok._fast_encode_trie = root

    # Preserve the original in case we need a reference
    orig_encode = tok.encode

    def fast_encode(text: str, add_bos: bool = True) -> list:
        text = tok.SP_PREFIX + text.replace(" ", tok.SP_PREFIX)
        ids: list = []
        if add_bos:
            ids.append(tok.BOS_ID)

        pos = 0
        n = len(text)
        trie = tok._fast_encode_trie
        id_map = tok.token_to_id

        while pos < n:
            # Walk the trie from `pos`, recording the longest matching
            # token end we hit.
            node = trie
            longest_tid = None
            longest_end = pos
            i = pos
            while i < n:
                c = text[i]
                child = node.get(c)
                if child is None:
                    break
                node = child
                i += 1
                end_tid = node.get("$$end")
                if end_tid is not None:
                    longest_tid = end_tid
                    longest_end = i
            if longest_tid is not None:
                ids.append(longest_tid)
                pos = longest_end
            else:
                # Byte fallback (matches original encoder)
                for b in text[pos].encode("utf-8"):
                    byte_tok = f"<0x{b:02X}>"
                    ids.append(id_map.get(byte_tok, 3))
                pos += 1
        return ids

    tok._orig_encode = orig_encode
    tok.encode = fast_encode
    print(f"[fast-tok] trie built over {len(tok.token_to_id)} tokens "
          f"in {time.time() - t0:.1f}s", flush=True)


# -------------------------------------------------------------
# RRF fusion
# -------------------------------------------------------------

def rrf_fuse(
    ranked_lists: List[List[Tuple[int, float]]],
    k_const: int = 60,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each `ranked_list` is a list of (doc_idx, orig_score) ordered best-
    first. Scores are IGNORED — RRF uses only rank. Returns a single
    list of (doc_idx, rrf_score), best-first, length `top_k`.

    k_const = 60 per Cormack et al. 2009 default. Insensitive to tuning
    in most setups.
    """
    fused: Dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (doc_idx, _) in enumerate(lst, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k_const + rank)
    items = sorted(fused.items(), key=lambda p: p[1], reverse=True)
    return items[:top_k]
