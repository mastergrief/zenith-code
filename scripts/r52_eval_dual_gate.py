"""R52.3 dual-gate eval — compare R52 KL-trained student to R51 MSE baseline.

Clone of scripts/r51_eval_dual_gate.py with the checkpoint path swapped
to r52_student_kl.pt. Same held-out corpus (seed=43, per_domain=20 = 120
prompts), same K_TOKENS=12, same gates (train-dist >= 0.80, off-dist >= 0.95).

Hypothesis: R51.5 baseline was train-dist 0.194 / off-dist 0.342 (BOTH FAIL).
R52 KL training targets the user-facing metric directly — expect meaningful
improvement. Binary decision: SHIP if both gates move up materially;
diagnose-and-null-commit if not.

Daemon-compatible. Assumes `m` and `tok` pre-bound.
"""

from __future__ import annotations

import random
import time

from calm.llm_computer.r51.install import (
    install_r51_student,
    load_student_from_checkpoint,
)
from calm.llm_computer.r51.prompt_bank import build_broad_corpus


CKPT_PATH = "calm/llm_computer/r51/checkpoints/r52_student_kl.pt"
R51_CKPT_PATH = "calm/llm_computer/r51/checkpoints/r51_student.pt"
TARGET_LAYER = 24
K_TOKENS = 12
PER_DOMAIN = 20
EVAL_SEED = 43
DOMAIN_ORDER = ["multi", "single", "trans", "code", "creative", "factual"]
TRAINING_DIST = {"multi", "single"}
OFF_DIST = {"trans", "code", "creative", "factual"}
GATE_TRAIN = 0.80
GATE_OFF = 0.95
R51_TRAIN = 0.194
R51_OFF = 0.342


def _generate_tokens(m, tok, prompt: str, k: int) -> list[int]:
    out = m.generate(prompt, tok, max_tokens=k, device="cuda",
                     stop_on_eos=False)
    return list(out["token_ids"])[:k]


def _prefix_match(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    matched = 0
    for i in range(n):
        if a[i] == b[i]:
            matched += 1
        else:
            break
    return matched


def _build_corpus() -> list[tuple[str, str]]:
    rng = random.Random(EVAL_SEED)
    prompts, counts = build_broad_corpus(rng, per_domain=PER_DOMAIN)
    print(f"[r52.3] held-out corpus: {len(prompts)} prompts, counts={counts}",
          flush=True)
    return prompts


def _emit_table(rows: list[tuple]) -> None:
    header = ("domain", "n", "exact-k-match", "mean-prefix-match")
    print(
        f"  {header[0]:<10} {header[1]:>4} {header[2]:>15} {header[3]:>20}",
        flush=True,
    )
    for row in rows:
        name, n, exact, mean_prefix = row
        print(
            f"  {name:<10} {n:>4} {exact:>10d}/{n:<4d} {mean_prefix:>20.4f}",
            flush=True,
        )


def main() -> None:
    assert "m" in globals(), "daemon contract: `m` must be pre-bound"
    assert "tok" in globals(), "daemon contract: `tok` must be pre-bound"

    print(f"[r52.3] loading R52 student from {CKPT_PATH}", flush=True)
    student = load_student_from_checkpoint(CKPT_PATH, device="cuda")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[r52.3] student: {n_params:,} params "
          f"({n_params / 1e6:.2f}M)", flush=True)

    prompts = _build_corpus()

    print(f"[r52.3] phase 1/2: baseline generations (k={K_TOKENS})",
          flush=True)
    t0 = time.time()
    baseline_tokens: list[list[int]] = []
    for i, (prompt, _label) in enumerate(prompts):
        tokens = _generate_tokens(m, tok, prompt, K_TOKENS)
        baseline_tokens.append(tokens)
        if (i + 1) % 20 == 0:
            print(f"  baseline {i + 1}/{len(prompts)} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    print(f"[r52.3] phase 2/2: installed generations "
          f"(L{TARGET_LAYER} -> R52 student)", flush=True)
    handle = install_r51_student(m, student, target_layer=TARGET_LAYER)
    installed_tokens: list[list[int]] = []
    t1 = time.time()
    try:
        for i, (prompt, _label) in enumerate(prompts):
            tokens = _generate_tokens(m, tok, prompt, K_TOKENS)
            installed_tokens.append(tokens)
            if (i + 1) % 20 == 0:
                print(f"  installed {i + 1}/{len(prompts)} "
                      f"({time.time() - t1:.1f}s)", flush=True)
    finally:
        handle.detach()
    print("[r52.3] student detached", flush=True)

    per_domain: dict[str, dict[str, float]] = {
        name: {"n": 0, "exact": 0, "prefix_sum": 0.0}
        for name in DOMAIN_ORDER
    }
    for (prompt, label), base, inst in zip(prompts, baseline_tokens,
                                           installed_tokens):
        pm = _prefix_match(base, inst)
        exact = int(pm == K_TOKENS and len(base) >= K_TOKENS
                    and len(inst) >= K_TOKENS)
        d = per_domain[label]
        d["n"] += 1
        d["exact"] += exact
        d["prefix_sum"] += pm / K_TOKENS

    print("", flush=True)
    print("=" * 72, flush=True)
    print("R52.3 DUAL-GATE VALIDATION (KL-divergence student)", flush=True)
    print("=" * 72, flush=True)
    print(
        f"held-out corpus: {len(prompts)} prompts "
        f"({PER_DOMAIN} per domain), seed={EVAL_SEED}, k={K_TOKENS}",
        flush=True,
    )
    print(f"checkpoint: {CKPT_PATH}", flush=True)
    print("", flush=True)
    print("Per-domain results:", flush=True)
    rows: list[tuple] = []
    for name in DOMAIN_ORDER:
        d = per_domain[name]
        n = d["n"]
        exact = int(d["exact"])
        mean_prefix = d["prefix_sum"] / n if n > 0 else 0.0
        rows.append((name, n, exact, mean_prefix))
    _emit_table(rows)

    train_prefix_sum = 0.0
    train_n = 0
    off_prefix_sum = 0.0
    off_n = 0
    for name in DOMAIN_ORDER:
        d = per_domain[name]
        if name in TRAINING_DIST:
            train_prefix_sum += d["prefix_sum"]
            train_n += d["n"]
        elif name in OFF_DIST:
            off_prefix_sum += d["prefix_sum"]
            off_n += d["n"]

    train_mean = train_prefix_sum / train_n if train_n > 0 else 0.0
    off_mean = off_prefix_sum / off_n if off_n > 0 else 0.0
    train_pass = train_mean >= GATE_TRAIN
    off_pass = off_mean >= GATE_OFF

    print("", flush=True)
    print("Gate summary:", flush=True)
    print(
        f"  training-distribution (multi + single, n={train_n}): "
        f"mean-prefix = {train_mean:.4f}  "
        f"[{'PASS' if train_pass else 'FAIL'} vs {GATE_TRAIN:.2f}]  "
        f"(R51.5 baseline: {R51_TRAIN:.3f})",
        flush=True,
    )
    print(
        f"  off-distribution (trans+code+creative+factual, n={off_n}): "
        f"mean-prefix = {off_mean:.4f}  "
        f"[{'PASS' if off_pass else 'FAIL'} vs {GATE_OFF:.2f}]  "
        f"(R51.5 baseline: {R51_OFF:.3f})",
        flush=True,
    )
    print("", flush=True)
    print(f"  R52 vs R51.5 delta: "
          f"train {train_mean - R51_TRAIN:+.4f}  "
          f"off-dist {off_mean - R51_OFF:+.4f}",
          flush=True)
    print("", flush=True)
    if train_pass and off_pass:
        verdict = "PASS BOTH GATES"
    elif train_pass and not off_pass:
        verdict = "TRAINING PASS, OFF-DIST FAIL"
    elif not train_pass and off_pass:
        verdict = "OFF-DIST PASS, TRAINING FAIL"
    else:
        verdict = "BOTH FAIL"
    print(f"  RESULT: {verdict}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
else:
    main()
