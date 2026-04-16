"""Auto-upgrade loop — CALM corrections → compiled weights → persistence.

Connects the existing pieces into one automatic pipeline:

  1. Model answers queries (some wrong)
  2. CALM verifies answers against 1002 backend functions
  3. Wrong answers logged as corrections
  4. End of session: corrections auto-compiled into substrate weights
  5. Save .pt
  6. Next session: errors permanently fixed, zero human intervention

Usage:
    engine = AutoUpgradeEngine(substrate, slots, store, substrate_path)

    # During session — user queries, CALM verifies
    answer = engine.query_with_verification("3 + 5")
    # If CALM corrects: auto-logged to knowledge store

    # End of session
    engine.commit()  # compile + install + save

    # Next session
    engine = AutoUpgradeEngine.load(substrate_path)
    # All previous corrections are in the weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from calm.llm_computer.persistent_knowledge import Correction, KnowledgeStore
from calm.llm_computer.hybrid_substrate import (
    HybridGroupedSmall2DTransformer, install_compiled_card_hybrid,
)


@dataclass
class QueryResult:
    """Result of a verified query."""
    prompt: str
    raw_answer: Optional[int]      # what the substrate produced
    verified_answer: Optional[int]  # what CALM says is correct (None = can't verify)
    was_correct: bool
    correction_applied: bool       # True if a new correction was logged


class AutoUpgradeEngine:
    """The self-improving loop: query → verify → correct → compile → persist.

    Holds the unified substrate + knowledge store. Queries go through
    the substrate's compiled card (dispatched_v4). Verification uses
    Python's own math functions (standing in for CALM's 1002 backends).
    Corrections accumulate in the knowledge store. `commit()` compiles
    them into the substrate's knowledge layer and saves.
    """

    def __init__(
        self,
        substrate: HybridGroupedSmall2DTransformer,
        card_tok_off: int,
        card_vocab: int,
        know_tok_off: int,
        know_vocab: int,
        know_ch_off: int,
        know_sh_off: int,
        know_ffn_off: int,
        know_layer_off: int,
        know_d_model: int,
        store: KnowledgeStore,
        save_path: Path,
    ):
        self.substrate = substrate
        self.card_tok_off = card_tok_off
        self.card_vocab = card_vocab
        self.know_tok_off = know_tok_off
        self.know_vocab = know_vocab
        self.know_ch_off = know_ch_off
        self.know_sh_off = know_sh_off
        self.know_ffn_off = know_ffn_off
        self.know_layer_off = know_layer_off
        self.know_d_model = know_d_model
        self.store = store
        self.save_path = save_path
        self.session_corrections: List[Correction] = []
        self.session_queries: List[QueryResult] = []

    def _verify_math(self, prompt: str) -> Optional[int]:
        """Stand-in for CALM verification — uses Python math to compute
        the correct answer. In production, this would be the full CALM
        engine with 1002 backend functions."""
        import re, math
        # Simple patterns CALM would catch
        m = re.match(r"(\d+)\s*\+\s*(\d+)", prompt)
        if m:
            return int(m.group(1)) + int(m.group(2))
        m = re.match(r"(\d+)\s*\*\s*(\d+)", prompt)
        if m:
            return int(m.group(1)) * int(m.group(2))
        m = re.match(r"(\d+)\s*-\s*(\d+)", prompt)
        if m:
            return int(m.group(1)) - int(m.group(2))
        m = re.match(r"gcd\((\d+),\s*(\d+)\)", prompt)
        if m:
            return math.gcd(int(m.group(1)), int(m.group(2)))
        m = re.match(r"factorial\((\d+)\)", prompt)
        if m:
            return math.factorial(int(m.group(1)))
        m = re.match(r"is_prime\((\d+)\)", prompt)
        if m:
            n = int(m.group(1))
            if n < 2:
                return 0
            return 1 if all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else 0
        return None

    def _query_knowledge(self, key: int) -> Optional[int]:
        """Query the knowledge layer for a previously-learned fact."""
        if key >= self.know_vocab or key < 0:
            return None
        x = torch.tensor([[key + self.know_tok_off]], dtype=torch.long)
        with torch.no_grad():
            logits = self.substrate(x)
        know_logits = logits[0, 0,
                            self.know_tok_off:self.know_tok_off + self.know_vocab]
        pred = int(know_logits.argmax().item())
        # Check if the knowledge layer actually matched (non-zero logit)
        if know_logits[pred].item() > 0.5:
            return pred
        return None

    def query_with_verification(self, prompt: str) -> QueryResult:
        """Query → verify → correct if wrong → log correction.

        The core loop: model tries to answer, CALM verifies, if wrong
        the correction is logged for end-of-session compilation.
        """
        # Step 1: compute verified answer via CALM stand-in
        verified = self._verify_math(prompt)
        if verified is None:
            return QueryResult(prompt, None, None, True, False)

        # Step 2: hash prompt to a key for knowledge lookup
        key = hash(prompt) % self.store.max_key

        # Step 3: check if we already know this (from previous corrections)
        known = self._query_knowledge(key)
        if known is not None and known == verified:
            result = QueryResult(prompt, known, verified, True, False)
            self.session_queries.append(result)
            return result

        # Step 4: if not known or wrong, log correction
        if known != verified:
            self.store.add_correction(key, verified)
            self.session_corrections.append(Correction(key, verified))
            result = QueryResult(prompt, known, verified,
                                known == verified, True)
        else:
            result = QueryResult(prompt, known, verified, True, False)

        self.session_queries.append(result)
        return result

    def commit(self) -> int:
        """End-of-session: compile all corrections into substrate weights,
        save .pt. Returns number of new corrections applied."""
        if not self.session_corrections:
            # Still save (might have other state changes)
            self._save()
            return 0

        # Rebuild knowledge model with ALL corrections (including previous).
        # Use a fixed min_d_ffn so the compiled model fits the pre-allocated
        # substrate slot regardless of how many corrections exist.
        know_model = self.store.build_recall_model(
            d_model=self.know_d_model, min_d_ffn=150,
        )

        # Zero out old knowledge layer weights, install new
        with torch.no_grad():
            l = self.know_layer_off
            self.substrate.W_qkv[l].weight.zero_()
            self.substrate.W_out[l].weight.zero_()
            self.substrate.ff_in[l].weight.zero_()
            self.substrate.ff_out[l].weight.zero_()
            # Re-zero the knowledge tok/head ranges
            self.substrate.tok.weight[
                self.know_tok_off:self.know_tok_off + self.know_vocab
            ].zero_()
            self.substrate.head.weight[
                :, self.know_ch_off:self.know_ch_off + self.know_d_model
            ].zero_()

        install_compiled_card_hybrid(
            self.substrate, know_model,
            ch_off=self.know_ch_off,
            sh_off=self.know_sh_off,
            ffn_off=self.know_ffn_off,
            tok_off=self.know_tok_off,
            layer_off=self.know_layer_off,
        )

        n = len(self.session_corrections)
        self._save()
        self.session_corrections.clear()
        return n

    def _save(self):
        """Save substrate + corrections to disk."""
        corr_path = self.save_path.with_suffix(".json")
        self.store.save_corrections(corr_path)
        torch.save(self.substrate.state_dict(), self.save_path)

    def report(self) -> str:
        """Session summary."""
        total = len(self.session_queries)
        correct = sum(1 for q in self.session_queries if q.was_correct)
        corrected = sum(1 for q in self.session_queries if q.correction_applied)
        return (f"queries={total} correct={correct} "
                f"corrections={corrected} "
                f"total_knowledge={len(self.store.corrections)}")
