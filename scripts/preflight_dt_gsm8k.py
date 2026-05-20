"""S0b1 preflight stats probe per codex audit `1779312982222-b1aff931`.

Mandate (verbatim): "Add a preflight stats receipt: p50/p95/max prompt chars,
chosen max_len, truncation count/rate, OOV chars/count, and train/val/test
split sizes."

Goal: before writing scripts/train_dt_gsm8k.py, prove the char-level
tokenizer/data contract holds on real GSM8k text. Counters:
- A length distribution that forces max_len > 512 (DT dense copy path)
- A char distribution with non-trivial OOV against the proposed local vocab
- Train/test split missing entirely

Run: `PYTHONPATH=. python3 scripts/preflight_dt_gsm8k.py`
Reads no checkpoint; writes no file. Stdout receipt only.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from statistics import median

from calm.llm_computer.gsm8k_tokenizer import (
    NORMALIZER_VERSION,
    SPECIAL_TOKENS,
    Gsm8kTokenizer,
    normalize_text,
)
from calm.llm_computer.gsm8k_tokenizer import _RESERVED_EXTRAS  # noqa: PLC2701


def fetch_split(split: str, max_rows: int = 8000) -> list[dict]:
    """Load GSM8k split via the `datasets` library (parquet backend, no
    per-row rate limits). Returns the same shape as the HF datasets-server
    rows-API converter used by `scripts/bv_step2/surface_gsm8k.py`.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows: list[dict] = []
    for i, r in enumerate(ds):
        if len(rows) >= max_rows:
            break
        gt = r["answer"]
        m = re.search(r"####\s*(-?[\d,]+)", gt)
        if not m:
            continue
        ans_str = m.group(1).replace(",", "").strip()
        try:
            expected = int(ans_str)
        except ValueError:
            continue
        rows.append({
            "id": f"gsm8k_{split}_{i}",
            "question": r["question"],
            "expected": expected,
            "answer_raw": gt,
        })
    return rows


def char_stats(samples: list[str]) -> dict:
    lens = sorted(len(s) for s in samples)
    n = len(lens)
    p50 = median(lens) if n else 0
    p95 = lens[int(n * 0.95)] if n else 0
    return {"n": n, "p50": p50, "p95": p95, "max": lens[-1] if n else 0,
            "min": lens[0] if n else 0}


def vocab_stats(samples: list[str]) -> Counter:
    c: Counter = Counter()
    for s in samples:
        c.update(s)
    return c


def main() -> int:
    print("=== S0b1 preflight stats — char-level DT on real GSM8k ===\n")

    print("[1/4] Fetching GSM8k train split (target ~7473 rows)...")
    train = fetch_split("train", max_rows=8000)
    print(f"      loaded {len(train)} train rows")

    print("[2/4] Fetching GSM8k test split (target 1319 rows)...")
    test = fetch_split("test", max_rows=2000)
    print(f"      loaded {len(test)} test rows")

    # Held-out val carved from train (last 10%, deterministic).
    val_frac = 0.10
    n_val = int(len(train) * val_frac)
    train_proper = train[:-n_val] if n_val else train
    val = train[-n_val:] if n_val else []
    print(f"[3/4] Proposed split: train={len(train_proper)}  val={len(val)}  test={len(test)}\n")

    # Per-string content for stats — target form per codex guardrail 3 is
    # final-integer-only: `<bos>question<sep>N<eos>`.
    def render_target(row: dict) -> str:
        return str(row["expected"])

    def render_full(row: dict) -> str:
        # Token-count proxy: question chars + sep + integer chars + bos + eos.
        return row["question"] + "|" + render_target(row)

    print("[4/4] Length stats (chars)\n")
    q_stats = char_stats([r["question"] for r in train])
    t_stats = char_stats([render_target(r) for r in train])
    full_stats = char_stats([render_full(r) for r in train])
    print(f"  question chars (train+val):  p50={q_stats['p50']}  p95={q_stats['p95']}  max={q_stats['max']}  min={q_stats['min']}")
    print(f"  target chars   (train+val):  p50={t_stats['p50']}  p95={t_stats['p95']}  max={t_stats['max']}  min={t_stats['min']}")
    print(f"  full (q+sep+t) (train+val):  p50={full_stats['p50']}  p95={full_stats['p95']}  max={full_stats['max']}")

    # Char vocab — train+val ONLY (codex audit `1779313349390`: test is an
    # OOV check, NOT a vocab source). Normalizer applied first so the
    # vocab matches the trained-time corpus.
    print(f"\n[vocab] normalizer policy: {NORMALIZER_VERSION} (smart quotes/dashes "
          "to ASCII, whitespace normalize, retain $ % ? : & ' \" etc.)")
    train_val = train_proper + val
    train_val_text = []
    for r in train_val:
        train_val_text.append(normalize_text(r["question"]))
        train_val_text.append(normalize_text(render_target(r)))
    vc_trainval = vocab_stats(train_val_text)
    print(f"  unique chars after normalize (train+val only): {len(vc_trainval)}")
    # Reserved-extras: justified a-priori, NOT from test set.
    declared_vocab_chars = set(vc_trainval.keys()) | _RESERVED_EXTRAS
    print(f"  declared vocab = train+val chars + reserved_extras "
          f"({sorted(_RESERVED_EXTRAS)}): {len(declared_vocab_chars)} chars")

    # OOV check: chars in test (post-normalize) that are NOT in train+val vocab.
    # Hard contract for the trainer: if this is non-zero, the normalizer
    # policy is incomplete and must be extended before locking the vocab.
    test_text = []
    for r in test:
        test_text.append(normalize_text(r["question"]))
        test_text.append(normalize_text(render_target(r)))
    vc_test = vocab_stats(test_text)
    test_oov = sorted(c for c in vc_test if c not in declared_vocab_chars)
    print(f"  OOV chars in test post-normalize vs declared vocab: {len(test_oov)}")
    fail = False
    if test_oov:
        repr_oov = [repr(c)[1:-1] for c in test_oov]
        print(f"    chars: {repr_oov}")
        print(f"    [FAIL] normalizer/reserved_extras policy incomplete; "
              f"extend declared vocab before locking")
        fail = True
    else:
        print("    [PASS] test post-normalize has no chars beyond declared vocab")

    # Compare against existing global char vocab in calm/hrm/data.py (audit only).
    GLOBAL_CHARS = set("0123456789+-*/()=., ;abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_><")
    oov_against_global = sorted(c for c in vc_trainval if c not in GLOBAL_CHARS)
    print(f"  audit: train+val vocab has {len(oov_against_global)} chars OOV vs calm/hrm/data.py:_CHARS")
    if oov_against_global:
        repr_oov = [repr(c)[1:-1] for c in oov_against_global]
        print(f"    chars: {repr_oov}")

    # Proposed local GSM8k vocab: bos/eos/pad/sep + declared chars (sorted).
    proposed_special = ["<pad>", "<bos>", "<eos>", "<sep>"]
    proposed_chars = sorted(declared_vocab_chars)
    print(f"\n  proposed local vocab (train+val + reserved_extras, normalize {NORMALIZER_VERSION}): "
          f"{len(proposed_special)} special + {len(proposed_chars)} chars = "
          f"{len(proposed_special) + len(proposed_chars)} tokens")

    # Truncation @ candidate max_len ceilings.
    full_lens = [len(s) + 3 for s in (render_full(r) for r in train + test)]  # +3 for bos+sep+eos
    for max_len in (128, 256, 384, 512, 768, 1024):
        truncated = sum(1 for L in full_lens if L > max_len)
        rate = truncated / max(len(full_lens), 1)
        print(f"  max_len={max_len:>4}: truncates {truncated}/{len(full_lens)} ({rate:.2%})")

    print("\n=== END preflight ===")
    # Hard-fail (nonzero exit) when test post-normalize is not covered by
    # declared vocab. Receipt says hard-fail, so CI / chained scripts get
    # an actual failure signal not a noisy zero-exit.
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
