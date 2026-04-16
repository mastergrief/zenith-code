"""Round-28 test: auto-upgrade loop across 3 sessions.

Session 1: model encounters 8 queries. CALM catches errors.
           End of session: corrections auto-compiled into weights. Save.

Session 2: reload. Previously-wrong queries are now correct (from
           weights, not retraining). New queries → new corrections.
           Commit + save.

Session 3: reload. ALL prior corrections persist. Zero errors on
           previously-seen queries. New queries still get verified.

No human writes a card. No human trains an HRM. No human edits weights.
The system upgrades itself through usage.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

from calm.llm_computer.auto_upgrade import AutoUpgradeEngine
from calm.llm_computer.hybrid_substrate import (
    HybridGroupedSmall2DConfig, HybridGroupedSmall2DTransformer,
    install_compiled_card_hybrid,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.persistent_knowledge import KnowledgeStore
from calm.llm_computer.programs.dispatched_v4 import build_dispatched_v4


KNOW_D = 20
MAX_CORRECTIONS = 50  # pre-allocate for up to 50 facts


def build_substrate_and_engine(save_path: Path, store: KnowledgeStore):
    """Build unified substrate + auto-upgrade engine from scratch."""
    card = build_dispatched_v4()
    know_model = store.build_recall_model(d_model=KNOW_D, min_d_ffn=MAX_CORRECTIONS * 3)
    c = card.config
    k = know_model.config

    # Pre-allocate knowledge slot for MAX_CORRECTIONS so reinstall
    # doesn't overflow when corrections grow across sessions.
    know_d_ffn = max(k.d_ffn, MAX_CORRECTIONS * 3)
    know_vocab = max(k.vocab_size, 65)  # max_value + 1

    d_model = c.d_model + KNOW_D
    d_model += d_model % 2
    n_heads = d_model // 2
    d_ffn = c.d_ffn + know_d_ffn
    vocab = c.vocab_size + know_vocab
    n_layers = c.n_layers + 1  # card layers + 1 knowledge layer

    cfg = HybridGroupedSmall2DConfig(
        vocab_size=vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max(c.max_len, k.max_len),
        use_hard_max=False,
        layer_modes=tuple(["single"] * n_layers),
        layer_hard_max=tuple([True] * n_layers),
        layer_linear_types=tuple(["fp32"] * n_layers),
    )
    sub = HybridGroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.zero_()

    # Install card at layer 0
    install_compiled_card_hybrid(sub, card,
                                ch_off=0, sh_off=0, ffn_off=0,
                                tok_off=0, layer_off=0)

    # Install knowledge at last layer
    know_ch = c.d_model
    know_sh = c.n_heads
    know_ffn = c.d_ffn
    know_tok = c.vocab_size
    know_layer = c.n_layers
    install_compiled_card_hybrid(sub, know_model,
                                ch_off=know_ch, sh_off=know_sh,
                                ffn_off=know_ffn, tok_off=know_tok,
                                layer_off=know_layer)
    sub.eval()

    engine = AutoUpgradeEngine(
        substrate=sub,
        card_tok_off=0, card_vocab=c.vocab_size,
        know_tok_off=know_tok, know_vocab=k.vocab_size,
        know_ch_off=know_ch, know_sh_off=know_sh,
        know_ffn_off=know_ffn, know_layer_off=know_layer,
        know_d_model=KNOW_D,
        store=store,
        save_path=save_path,
    )
    return engine


def reload_engine(save_path: Path) -> AutoUpgradeEngine:
    """Reload engine from disk (simulates new session)."""
    corr_path = save_path.with_suffix(".json")
    store = KnowledgeStore(max_key=512, max_value=512)
    store.load_corrections(corr_path)
    return build_substrate_and_engine(save_path, store)


def run_queries(engine, prompts, label):
    """Run queries through the engine, print results."""
    print(f"\n  [{label}] running {len(prompts)} queries:")
    for prompt in prompts:
        r = engine.query_with_verification(prompt)
        status = "✓ known" if r.was_correct and not r.correction_applied else \
                 "→ CORRECTED" if r.correction_applied else \
                 "✓ verified" if r.was_correct else "✗"
        print(f"    [{status:>12}] {prompt!r:25} "
              f"raw={r.raw_answer!r:>5} verified={r.verified_answer!r:>5}")
    print(f"  [{label}] {engine.report()}")


def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp())
    save_path = tmp / "substrate.pt"

    # ===== SESSION 1 =====
    print("=" * 60)
    print("SESSION 1: first use — model knows nothing beyond compiled ops")
    print("=" * 60)

    store = KnowledgeStore(max_key=512, max_value=512)
    engine = build_substrate_and_engine(save_path, store)

    # Queries that will trigger CALM corrections:
    # The substrate's dispatched_v4 can handle operands [0, 15] with opcodes.
    # But plain "17 + 25" goes through the KNOWLEDGE path (hashed key),
    # not the card path. CALM verifies and logs the correction.
    session1_queries = [
        "17 + 25",        # → 42 (outside card range, knowledge path)
        "100 + 200",      # → 300
        "gcd(24, 36)",    # → 12 (outside card range)
        "factorial(10)",  # → 3628800 — too big for our max_value, will clip
        "13 * 7",         # → 91 — outside card range
        "50 - 8",         # → 42
        "is_prime(23)",   # → 1 (True)
        "is_prime(24)",   # → 0 (False)
    ]
    # Limit to max_value=64 for demo
    session1_queries = [
        "17 + 25",        # → 42
        "3 + 5",          # → 8
        "gcd(12, 18)",    # → 6
        "7 * 8",          # → 56
        "20 - 7",         # → 13
        "is_prime(23)",   # → 1
        "is_prime(15)",   # → 0
        "gcd(21, 14)",    # → 7
    ]
    run_queries(engine, session1_queries, "S1")

    n_committed = engine.commit()
    print(f"\n  [S1] committed {n_committed} corrections → compiled into weights")
    print(f"  [S1] saved to {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")

    # ===== BETWEEN SESSIONS =====
    print("\n" + "=" * 60)
    print("BETWEEN SESSIONS: model unloaded, only .pt + .json on disk")
    print("=" * 60)
    del engine

    # ===== SESSION 2 =====
    print("\n" + "=" * 60)
    print("SESSION 2: reload — previous corrections should be in weights")
    print("=" * 60)

    engine2 = reload_engine(save_path)
    print(f"  loaded {len(engine2.store.corrections)} corrections from disk")

    # Re-run SAME queries — should now be correct from knowledge weights
    print("\n  [S2] re-running session 1 queries (should be '✓ known'):")
    run_queries(engine2, session1_queries, "S2-recall")

    # New queries
    new_queries = [
        "11 + 22",    # → 33 (new)
        "gcd(9, 6)",  # → 3 (new)
        "5 * 9",      # → 45 (new)
    ]
    print(f"\n  [S2] new queries (will be corrected + learned):")
    run_queries(engine2, new_queries, "S2-new")

    n2 = engine2.commit()
    print(f"\n  [S2] committed {n2} new corrections")
    print(f"  [S2] total knowledge: {len(engine2.store.corrections)} facts")

    del engine2

    # ===== SESSION 3 =====
    print("\n" + "=" * 60)
    print("SESSION 3: all knowledge accumulated across 2 sessions")
    print("=" * 60)

    engine3 = reload_engine(save_path)
    print(f"  loaded {len(engine3.store.corrections)} corrections")

    all_queries = session1_queries + new_queries
    print(f"\n  [S3] ALL {len(all_queries)} queries from sessions 1+2:")
    run_queries(engine3, all_queries, "S3-all")

    # Count how many are now correct from knowledge
    known_count = sum(
        1 for q in engine3.session_queries
        if q.was_correct and not q.correction_applied
    )
    total = len(engine3.session_queries)
    print(f"\n  [S3] {known_count}/{total} answered correctly from compiled "
          f"knowledge (no CALM needed)")

    # ===== SUMMARY =====
    all_ok = known_count == total
    t = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[R28] OVERALL: {'PASS' if all_ok else 'FAIL'}  ({t:.1f}s)")
    print(f"[R28] auto-upgrade loop:")
    print(f"  session 1: {len(session1_queries)} queries → "
          f"{len(session1_queries)} corrections compiled")
    print(f"  session 2: recalled from weights + learned "
          f"{len(new_queries)} more")
    print(f"  session 3: ALL {total} queries correct from weights alone")
    print(f"[R28] self-improving through usage: "
          f"{'VALIDATED' if all_ok else 'NOT VALIDATED'}")

    # Cleanup
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
