"""Persistent knowledge — corrections → weight updates → cross-session memory.

The continuous learning loop for the substrate:

  Session N:
    1. Model encounters query "gcd(12, 18)" → gets it wrong
    2. CALM corrects → logs (query_key, correct_value) pair
    3. End of session: compile corrections into substrate weights

  Between sessions:
    4. For each correction: write a new token embedding row encoding
       the query key and correct value at reserved channels
    5. Install a compiled "recall" program that does LookUpExact on
       the key channels → retrieves the value
    6. Save substrate .pt

  Session N+1:
    7. Load substrate → corrections are in the weights
    8. Forward on same query → compiled recall produces correct answer
    9. Zero retraining, zero gradient descent

Architecture:
  * Token embedding rows [KNOWLEDGE_BASE_START, ...] hold facts.
    Each row = one fact: key at CH_FACT_KEY, value at CH_FACT_VALUE.
  * A compiled "recall" sub-program at a designated layer reads the
    query (from CH_OWN at the query position), matches against
    stored keys via step functions, and writes the matching value
    to CH_RECALL_RESULT.

For this MVP: facts are (integer_key → integer_value) pairs, stored as
step-function lookup in the FFN (same pattern as dispatched.py's GCD).
Scalable to arbitrary key spaces via LookUpExact parabolic keys.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class Correction:
    """One fact learned from a correction."""
    query_key: int      # the input key (e.g., hash of "gcd(12,18)")
    correct_value: int  # the verified correct answer


@dataclass
class KnowledgeStore:
    """Manages corrections → weight updates → persistence.

    Holds a list of corrections and can compile them into a substrate's
    weights. Each correction becomes a (key → value) entry retrievable
    by forward pass.
    """
    corrections: List[Correction] = field(default_factory=list)
    max_key: int = 64      # key range [0, max_key)
    max_value: int = 64    # value range [0, max_value)

    def add_correction(self, query_key: int, correct_value: int):
        # Deduplicate: latest correction for a key wins
        self.corrections = [
            c for c in self.corrections if c.query_key != query_key
        ]
        self.corrections.append(Correction(query_key, correct_value))

    def save_corrections(self, path: Path):
        """Save corrections to JSON for cross-session persistence."""
        data = [{"key": c.query_key, "value": c.correct_value}
                for c in self.corrections]
        path.write_text(json.dumps(data))

    def load_corrections(self, path: Path):
        """Load corrections from a previous session."""
        data = json.loads(path.read_text())
        for item in data:
            self.add_correction(item["key"], item["value"])

    def build_recall_model(self, d_model: int = 16,
                           max_len: int = 4) -> Small2DTransformer:
        """Compile all corrections into a Small2DTransformer that does
        key → value lookup via step-function dispatch.

        Input: [query_key] at position 0.
        Output: argmax at position 0 = correct_value (or 0 if no match).

        Architecture:
          CH_OWN (0): query key from token embedding
          CH_BIAS (1): constant 1
          CH_MATCHED_VALUE (2+k): gated step for key k

        For each correction (key_k, value_k): a pair of ReGLU neurons
        implements indicator(CH_OWN == key_k) and gates the value.
        Head maps each matched channel to its value slot.
        """
        vocab = max(self.max_key, self.max_value) + 1
        n_heads = d_model // 2

        # Channel layout
        CH_OWN = 0
        CH_BIAS = 1
        CH_MATCH_BASE = 2  # one channel per stored correction

        n_corrections = len(self.corrections)
        needed_channels = CH_MATCH_BASE + n_corrections
        if needed_channels > d_model:
            d_model = needed_channels + (needed_channels % 2)
            n_heads = d_model // 2

        graph = GateGraph(vocab_size=vocab)
        graph.add(TokenEmbed(
            name="tok",
            entries=[(k, CH_OWN, float(k)) for k in range(vocab)],
        ))
        graph.add(PosEmbed(
            name="bias",
            entries=[(p, CH_BIAS, 1.0) for p in range(max_len)],
        ))

        # For each correction: indicator(CH_OWN == key_k) → channel
        # indicator(x == k) = step_k(x) - step_{k+1}(x)
        #   = [ReLU(x-k+1) - ReLU(x-k)] - [ReLU(x-k) - ReLU(x-k-1)]
        #   = ReLU(x-k+1) - 2*ReLU(x-k) + ReLU(x-k-1)
        # 3 ReGLU neurons per correction (coefs +1, -2, +1)
        for i, corr in enumerate(self.corrections):
            ch = CH_MATCH_BASE + i
            k = corr.query_key
            # +ReLU(own - k + 1)
            graph.add(ReGLU(
                name=f"match_{i}_a", layer=0,
                gate=[(CH_OWN, 1.0), (CH_BIAS, -(k - 1))],
                val=[(CH_BIAS, 1.0)],
                output_channel=ch,
                output_coef=1.0,
            ))
            # -2*ReLU(own - k)
            graph.add(ReGLU(
                name=f"match_{i}_b", layer=0,
                gate=[(CH_OWN, 1.0), (CH_BIAS, -k)],
                val=[(CH_BIAS, 1.0)],
                output_channel=ch,
                output_coef=-2.0,
            ))
            # +ReLU(own - k - 1)
            graph.add(ReGLU(
                name=f"match_{i}_c", layer=0,
                gate=[(CH_OWN, 1.0), (CH_BIAS, -(k + 1))],
                val=[(CH_BIAS, 1.0)],
                output_channel=ch,
                output_coef=1.0,
            ))

        # Head: when match channel i fires (=1), contribute to slot = value_i
        head_entries = []
        for i, corr in enumerate(self.corrections):
            ch = CH_MATCH_BASE + i
            head_entries.append((corr.correct_value, ch, 1.0))
        graph.add(LinearHead(name="recall_head", entries=head_entries))

        d_ffn = max(4, 3 * n_corrections)
        return compile_program(
            graph, d_model=d_model, n_heads=n_heads, n_layers=1,
            d_ffn=d_ffn, max_len=max_len, vocab_size=vocab,
        )

    def query(self, model: Small2DTransformer, key: int) -> int:
        """Query the recall model for a key. Returns argmax slot."""
        x = torch.tensor([[key]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        return int(logits[0, 0].argmax().item())


if __name__ == "__main__":
    import time

    print("[R26] === SIMULATED CONTINUOUS LEARNING LOOP ===\n")

    # ---- SESSION 1: encounter errors, log corrections ----
    print("[session 1] model encounters queries and gets some wrong...")
    store = KnowledgeStore(max_key=64, max_value=64)

    # Simulate: CALM catches errors and logs corrections
    corrections_from_calm = [
        (7, 6),    # "gcd(42, 18)" → hashed to key 7, correct value 6
        (12, 24),  # "factorial(4)" → key 12, value 24
        (23, 1),   # "is_prime(23)" → key 23, value 1 (True)
        (15, 0),   # "is_prime(15)" → key 15, value 0 (False)
        (50, 42),  # "17 + 25" → key 50, value 42
    ]
    for key, value in corrections_from_calm:
        store.add_correction(key, value)
        print(f"  logged: key={key} → value={value}")

    print(f"\n[session 1] compiling {len(store.corrections)} corrections "
          f"into recall model...")
    model_v1 = store.build_recall_model()
    print(f"  d_model={model_v1.config.d_model}, "
          f"params={model_v1.param_count()}")

    # Verify: all corrections retrievable
    print("\n[session 1] verifying all corrections in model:")
    v1_ok = 0
    for key, expected in corrections_from_calm:
        got = store.query(model_v1, key)
        mark = "✓" if got == expected else "✗"
        print(f"  [{mark}] key={key} → {got} (expected {expected})")
        v1_ok += (got == expected)
    print(f"  v1 recall: {v1_ok}/{len(corrections_from_calm)}")

    # Verify: unknown keys return 0 (no match)
    print("\n[session 1] unknown keys return 0 (no match):")
    for key in [0, 1, 63]:
        got = store.query(model_v1, key)
        print(f"  key={key} → {got} (expected 0)")

    # Save corrections + model
    tmp = Path(tempfile.mkdtemp())
    corrections_path = tmp / "corrections.json"
    model_path = tmp / "substrate_v1.pt"
    store.save_corrections(corrections_path)
    torch.save(model_v1.state_dict(), model_path)
    print(f"\n[session 1] saved corrections to {corrections_path}")
    print(f"[session 1] saved model to {model_path} "
          f"({model_path.stat().st_size} bytes)")

    # ---- BETWEEN SESSIONS: simulate time passing ----
    print("\n" + "=" * 60)
    print("[between sessions] model unloaded, state on disk only")
    print("=" * 60)
    del model_v1, store

    # ---- SESSION 2: load + verify persistence ----
    print("\n[session 2] loading corrections + model from disk...")
    store2 = KnowledgeStore(max_key=64, max_value=64)
    store2.load_corrections(corrections_path)
    print(f"  loaded {len(store2.corrections)} corrections")

    model_v2 = store2.build_recall_model()
    sd = torch.load(model_path, weights_only=True)
    model_v2.load_state_dict(sd)
    print(f"  model loaded, params={model_v2.param_count()}")

    print("\n[session 2] verifying ALL session-1 corrections persist:")
    v2_ok = 0
    for key, expected in corrections_from_calm:
        got = store2.query(model_v2, key)
        mark = "✓" if got == expected else "✗"
        print(f"  [{mark}] key={key} → {got} (expected {expected})")
        v2_ok += (got == expected)

    # ---- SESSION 2: add MORE corrections ----
    new_corrections = [
        (30, 15),  # new fact
        (7, 3),    # OVERRIDE: key 7 was 6, now corrected to 3
    ]
    print(f"\n[session 2] adding {len(new_corrections)} new corrections "
          f"(including 1 override)...")
    for key, value in new_corrections:
        store2.add_correction(key, value)
        print(f"  logged: key={key} → value={value}")

    model_v3 = store2.build_recall_model()
    print(f"  rebuilt model with {len(store2.corrections)} total corrections")

    print("\n[session 2] verifying updated knowledge:")
    all_expected = {k: v for k, v in corrections_from_calm}
    all_expected.update({k: v for k, v in new_corrections})  # override key 7
    v3_ok = 0
    for key, expected in sorted(all_expected.items()):
        got = store2.query(model_v3, key)
        mark = "✓" if got == expected else "✗"
        note = " (overridden)" if key == 7 else ""
        print(f"  [{mark}] key={key} → {got} (expected {expected}){note}")
        v3_ok += (got == expected)

    # Save v3
    model_v3_path = tmp / "substrate_v3.pt"
    store2.save_corrections(corrections_path)
    torch.save(model_v3.state_dict(), model_v3_path)

    # ---- SUMMARY ----
    all_ok = (v1_ok == len(corrections_from_calm)
              and v2_ok == len(corrections_from_calm)
              and v3_ok == len(all_expected))
    print(f"\n[R26] SUMMARY:")
    print(f"  session 1 recall:       {v1_ok}/{len(corrections_from_calm)}")
    print(f"  session 2 persistence:  {v2_ok}/{len(corrections_from_calm)}")
    print(f"  session 2 + new facts:  {v3_ok}/{len(all_expected)}")
    print(f"  override (key 7: 6→3):  "
          f"{'PASS' if store2.query(model_v3, 7) == 3 else 'FAIL'}")
    print(f"\n[R26] OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print(f"[R26] continuous learning via weight updates:")
    print(f"[R26]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    if all_ok:
        print("[R26]   corrections compiled into weights, persisted across")
        print("[R26]   sessions, overrides replace old facts, zero retraining")

    # Cleanup
    corrections_path.unlink()
    model_path.unlink()
    model_v3_path.unlink()
    tmp.rmdir()
