"""R51.5 dual-gate validation: task preservation for the R51 student on
training-distribution AND off-distribution held-out prompts.

Daemon-compatible. Assumes `m` (GemmaSubstrate) and `tok` (GemmaTokenizer)
are pre-bound in globals by `bin/gemma_daemon.py`. Does NOT load Gemma.

Gate design (from R51 hypothesis in SESSION_HANDOFF.md):
    training-dist (multi + single):  mean prefix match >= 0.80
    off-dist (trans + code + creative + factual):  mean prefix match >= 0.95

Held-out prompts are drawn from `build_broad_corpus(rng=Random(43), per_domain=20)`.
Seed 43 != training's seed 42; R51.1 used `per_domain=500`, so the 20-per-domain
corpus here is a DIFFERENT random draw even before the seed change (the RNG
state after 500 draws in a pool differs from the first 20). Per-domain dedup
inside the sampler + the seed gap gives a clean held-out set.

Evaluation loop:
    1. Run N_BASELINE baseline generations (Gemma unmodified) and record
       the K-token sequences.
    2. Install the student ONCE via `install_r51_student(m, student)`.
    3. Run N_INSTALLED generations (Gemma with L24 replaced) and record
       the same K-token sequences.
    4. Detach install and compare per-prompt.

Install-once-then-detach: cheaper than install/detach per prompt and the
student is stateless (no per-prompt memory), so the pairing stays valid.
Baselines first so any install-side bug can't contaminate them.
"""

from __future__ import annotations

import random
import time

from calm.llm_computer.r51.install import (
    install_r51_student,
    load_student_from_checkpoint,
)
from calm.llm_computer.r51.prompt_bank import build_broad_corpus


CKPT_PATH = "calm/llm_computer/r51/checkpoints/r51_student.pt"
TARGET_LAYER = 24
K_TOKENS = 12
PER_DOMAIN = 20
EVAL_SEED = 43
DOMAIN_ORDER = ["multi", "single", "trans", "code", "creative", "factual"]
TRAINING_DIST = {"multi", "single"}
OFF_DIST = {"trans", "code", "creative", "factual"}
GATE_TRAIN = 0.80
GATE_OFF = 0.95


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
    print(f"[r51.5] held-out corpus: {len(prompts)} prompts, counts={counts}",
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

    print(f"[r51.5] loading student from {CKPT_PATH}", flush=True)
    student = load_student_from_checkpoint(CKPT_PATH, device="cuda")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[r51.5] student loaded: {n_params:,} params "
          f"({n_params / 1e6:.2f}M)", flush=True)

    prompts = _build_corpus()

    print(f"[r51.5] phase 1/2: baseline generations (k={K_TOKENS})",
          flush=True)
    t0 = time.time()
    baseline_tokens: list[list[int]] = []
    for i, (prompt, _label) in enumerate(prompts):
        tokens = _generate_tokens(m, tok, prompt, K_TOKENS)
        baseline_tokens.append(tokens)
        if (i + 1) % 20 == 0:
            print(f"  baseline {i + 1}/{len(prompts)} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    print(f"[r51.5] phase 2/2: installed generations (L{TARGET_LAYER} -> student)",
          flush=True)
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
    print("[r51.5] student detached", flush=True)

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
    print("R51.5 DUAL-GATE VALIDATION", flush=True)
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
        f"[{'PASS' if train_pass else 'FAIL'} vs {GATE_TRAIN:.2f}]",
        flush=True,
    )
    print(
        f"  off-distribution (trans+code+creative+factual, n={off_n}): "
        f"mean-prefix = {off_mean:.4f}  "
        f"[{'PASS' if off_pass else 'FAIL'} vs {GATE_OFF:.2f}]",
        flush=True,
    )
    if train_pass and off_pass:
        verdict = "PASS BOTH GATES"
    elif train_pass and not off_pass:
        verdict = "TRAINING PASS, OFF-DIST FAIL"
    elif not train_pass and off_pass:
        verdict = "OFF-DIST PASS, TRAINING FAIL"
    else:
        verdict = "BOTH FAIL"
    print("", flush=True)
    print(f"  RESULT: {verdict}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
else:
    main()
