"""Fast Weights vs Vanilla Small2DTransformer on Associative Recall.

Round 1 of the runtime-weight-addition research track. Binary decision:
does Schlag-style asymmetric fast weights close the associative-recall
gap on held-out KV tokens?

Task: sequences of form [k_1 v_1 k_2 v_2 ... k_N v_N SEP k_q] with
target v_q at the final position. Keys and values drawn from a training
pool; eval uses a disjoint held-out pool. This tests whether the model
can bind NOVEL key-value pairs introduced mid-sequence, not just memorize
the train distribution.

Decision rule (printed at the end):
  PASS if fast-weights 3-pair recall >= 70%
  FAIL if fast-weights 3-pair recall <  50%
  INCONCLUSIVE if in [50%, 70%) — one hyperparam sweep allowed before
  final call.

Runtime: ~1-3 min on CPU (2 variants × 3 n_pairs × 100 epochs).
"""

from __future__ import annotations

import random
import sys
import time

import torch
import torch.nn.functional as F

from calm.llm_computer.fast_weights import (
    FastWeightConfig, FastWeightSmall2DTransformer,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# ----- task / model config -----
#
# Generalization is to novel *sequences* (fresh k→v bindings drawn from
# the pool), NOT to novel tokens. Splitting tokens between train/eval is
# the wrong hold-out: eval tokens' embeddings + head entries never get
# gradient, so eval accuracy is 0% for any model. The right hold-out is
# a fresh RNG seed producing sequences that never appeared in training,
# drawn from the same token pool. With a 256-token pool and 3-pair
# sequences, the sequence space is ~10^9 — memorization is not a concern.
VOCAB_SIZE = 257   # 0 (SEP) + 256 content tokens
SEP_TOKEN = 0
TOKEN_POOL = list(range(1, 257))    # 256 tokens, shared train+eval
MAX_LEN = 64

D_MODEL = 64
N_HEADS = 32          # d_head = 2 (substrate invariant)
N_LAYERS = 2
D_FFN = 128

# ----- training / eval config -----
EPOCHS = 100
BATCH_SIZE = 32
SEQUENCES_PER_EPOCH = 1024
STEPS_PER_EPOCH = SEQUENCES_PER_EPOCH // BATCH_SIZE
LR = 1e-3
N_EVAL = 1024   # must be multiple of BATCH_SIZE

# ----- fast-weights hyperparams -----
LAMBDA_DECAY = 0.95
ETA_WRITE = 0.5


def build_sequence(n_pairs: int, token_pool: list[int], rng: random.Random):
    """One KV-recall sequence: [k1 v1 ... kN vN SEP k_q] → target v_q."""
    keys = rng.sample(token_pool, n_pairs)           # distinct keys
    values = [rng.choice(token_pool) for _ in range(n_pairs)]
    query_idx = rng.randrange(n_pairs)
    query_key = keys[query_idx]
    target = values[query_idx]
    seq: list[int] = []
    for k, v in zip(keys, values):
        seq.extend([k, v])
    seq.extend([SEP_TOKEN, query_key])
    return seq, target


def build_batch(n_pairs: int, token_pool: list[int], batch_size: int,
                rng: random.Random):
    seqs, targets = [], []
    for _ in range(batch_size):
        s, t = build_sequence(n_pairs, token_pool, rng)
        seqs.append(s)
        targets.append(t)
    return (torch.tensor(seqs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def train_model(model, n_pairs: int, label: str):
    """Train on KV-recall with fixed n_pairs. Uses softmax attention."""
    rng = random.Random(42 + n_pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    orig_hard_max = model.config.use_hard_max
    model.config.use_hard_max = False
    model.train()

    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss, total_correct, total = 0.0, 0, 0
        for _ in range(STEPS_PER_EPOCH):
            x, target = build_batch(n_pairs, TOKEN_POOL, BATCH_SIZE, rng)
            logits = model(x)[:, -1, :]          # predict at final position
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(-1) == target).sum().item()
            total += x.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"    [{label} n={n_pairs}] epoch {epoch+1:3d}/{EPOCHS}: "
                f"loss={total_loss/total:.4f}  train_acc={total_correct/total:.2%}  "
                f"{elapsed:5.1f}s",
                flush=True,
            )

    model.config.use_hard_max = orig_hard_max
    model.eval()


def eval_recall(model, n_pairs: int, token_pool: list[int],
                n_eval: int = N_EVAL, seed: int = 9999):
    """Argmax-at-final-position accuracy on held-out pool. Softmax attention."""
    rng = random.Random(seed + n_pairs)
    orig_hard_max = model.config.use_hard_max
    model.config.use_hard_max = False
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_eval // BATCH_SIZE):
            x, target = build_batch(n_pairs, token_pool, BATCH_SIZE, rng)
            preds = model(x)[:, -1, :].argmax(-1)
            correct += (preds == target).sum().item()
            total += x.size(0)
    model.config.use_hard_max = orig_hard_max
    return correct, total


def make_vanilla():
    cfg = Small2DConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ffn=D_FFN, max_len=MAX_LEN, use_hard_max=False,
    )
    return Small2DTransformer(cfg)


def make_fast():
    cfg = FastWeightConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ffn=D_FFN, max_len=MAX_LEN, use_hard_max=False,
        lambda_decay=LAMBDA_DECAY, eta_write=ETA_WRITE, use_fast_weights=True,
    )
    return FastWeightSmall2DTransformer(cfg)


def run_variant(make_model, label: str, n_pairs_list: list[int]):
    results = {}
    for n_pairs in n_pairs_list:
        print(f"\n  training {label} with n_pairs={n_pairs}...", flush=True)
        torch.manual_seed(0)
        model = make_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    {n_params} parameters", flush=True)
        train_model(model, n_pairs, label)
        correct, total = eval_recall(model, n_pairs, TOKEN_POOL)
        results[n_pairs] = (correct, total)
        print(f"    eval (held-out pool): {correct}/{total} = {correct/total:.1%}",
              flush=True)
    return results


def decision_line(fast_3pair_rate: float) -> str:
    if fast_3pair_rate >= 0.70:
        return f"DECISION: PASS  (fast-weights 3-pair = {fast_3pair_rate:.1%} >= 70%)"
    if fast_3pair_rate < 0.50:
        return f"DECISION: FAIL  (fast-weights 3-pair = {fast_3pair_rate:.1%} < 50%)"
    return (f"DECISION: INCONCLUSIVE  (fast-weights 3-pair = "
            f"{fast_3pair_rate:.1%} in [50%, 70%))")


def main():
    print("=== Round 1: Fast Weights vs Vanilla on Associative Recall ===")
    print(f"  vocab={VOCAB_SIZE}  d_model={D_MODEL}  n_heads={N_HEADS}  "
          f"d_head={D_MODEL//N_HEADS}  n_layers={N_LAYERS}  d_ffn={D_FFN}")
    print(f"  token pool: [{TOKEN_POOL[0]}, {TOKEN_POOL[-1]+1})  "
          f"(shared train+eval; hold-out is fresh sequences, not tokens)")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  "
          f"seq/epoch={SEQUENCES_PER_EPOCH}  lr={LR}")
    print(f"  fast weights: lambda_decay={LAMBDA_DECAY}  eta_write={ETA_WRITE}")
    sys.stdout.flush()

    n_pairs_list = [3, 5, 10]

    print("\n[1/2] vanilla Small2DTransformer")
    vanilla = run_variant(make_vanilla, "vanilla", n_pairs_list)

    print("\n[2/2] FastWeightSmall2DTransformer")
    fast = run_variant(make_fast, "fast", n_pairs_list)

    print("\n" + "=" * 60)
    print("Round 1 Results")
    print("=" * 60)
    print("\n  n_pairs  vanilla       fast-weights   delta")
    print("  -------  ----------    ----------     --------")
    for n in n_pairs_list:
        cv, tv = vanilla[n]
        cf, tf = fast[n]
        rv = cv / tv
        rf = cf / tf
        delta = (rf - rv) * 100
        print(f"  {n:>7}  {cv:>4}/{tv} = {rv:5.1%}  "
              f"{cf:>4}/{tf} = {rf:5.1%}  {delta:+6.1f} pp")

    print("\n" + "=" * 60)
    fast_3pair_rate = fast[3][0] / fast[3][1]
    print(decision_line(fast_3pair_rate))
    print("=" * 60)


if __name__ == "__main__":
    main()
