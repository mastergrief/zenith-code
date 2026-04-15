"""SubstrateHRMLM v2 — main thinking engine of the substrate.

Builds on hybrid v1's PARTIAL result (LM +9% transfer, HRM 0%) by fixing
all three diagnoses + applying every D2/D3/D5 substrate extension:

  v1 → v2:
  * HRM template variety: nl_data only → pooled nl + word + gsm + multi20
    (~50+ phrasings vs 15) — applies session-26 multi20 lesson.
  * Token rebalance: HRM examples upsampled 4× to neutralize LM token
    dominance (LM ~500 tok/ex, HRM ~30 tok/ex → effective 1:1 gradient).
  * Mode-loss weighting: HRM-mode loss * 4× during training.
  * D3 mixed geometry per layer: Euclidean / hyperbolic / lattice mix.
  * D5 recurrent substrate: HRM mode runs at higher iteration count.
  * D2 trace emission for eval inspection.
  * Model size: 1.25M (v1) → ~3-4M params (d_model=128, n_layers=4).

Training: ~25-40 min CPU. Eval per HRM template family separately so we
see where the model strengthens vs where it still struggles.

Decision criteria:
  PASS: HRM correctness >= 50% on at least one template family AND
        LM ppl <= 450 (no major LM degradation).
  PARTIAL: HRM correctness 20-50% on best family.
  FAIL: HRM <20% on all families (architecture/curriculum still wrong).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from calm.expression import safe_eval
from calm.hrm.gsm_data import GSMDataGenerator
from calm.hrm.multi20_data import Multi20Generator
from calm.hrm.nl_data import NLMathDataGenerator
from calm.hrm.word_data import WordProblemGenerator
from calm.llm_computer.combined_substrate import (
    CombinedConfig, CombinedSmall2DTransformer,
)
from calm.llm_computer.computation_trace import make_trace_collector
from calm.llm_computer.interpret import interpret
from calm.llm_computer.parse import parse_expression


REPO_ROOT = Path(__file__).resolve().parent.parent
LM_DATA_PATHS = [
    REPO_ROOT / "agents" / "distill" / "data" / "claude_reasoning.jsonl",
    REPO_ROOT / "agents" / "distill" / "data" / "coding_reasoning_claude.jsonl",
]
CKPT_DIR = REPO_ROOT / "calm" / "llm_computer" / "checkpoints"


# Special tokens
LM_MODE = "<|lm|>"
HRM_MODE = "<|hrm|>"
SYS_TOKEN = "<|sys|>"
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|asst|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [LM_MODE, HRM_MODE, SYS_TOKEN, USER_TOKEN, ASST_TOKEN,
                  EOS_TOKEN, PAD_TOKEN]


@dataclass
class HRMExample:
    """Standardized HRM example: question, expression, answer."""
    family: str          # nl / word / gsm / multi20
    question: str
    expression: str
    answer: str


# ----- Corpus assembly -----

def load_lm_examples(jsonl_paths) -> list[tuple[str, int, str]]:
    """Returns (text, asst_start_char, mode) tuples for LM training."""
    out = []
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                msgs = ex.get("messages", [])
                parts = [LM_MODE]
                asst_start = -1
                for m in msgs:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        parts.append(f"{SYS_TOKEN}{content}")
                    elif role == "user":
                        parts.append(f"{USER_TOKEN}{content}")
                    elif role == "assistant":
                        pre = "".join(parts)
                        asst_start = len(pre) + len(ASST_TOKEN)
                        parts.append(f"{ASST_TOKEN}{content}{EOS_TOKEN}")
                if asst_start > 0:
                    out.append(("".join(parts), asst_start, "lm"))
    return out


def collect_hrm_examples(per_family: int = 600, seed: int = 42) -> list[HRMExample]:
    """Pool NL, word, GSM, multi20. Standardize fields per family."""
    out = []
    # nl_data: question, expression, answer
    for p in NLMathDataGenerator(seed=seed).generate(n=per_family):
        out.append(HRMExample("nl", p.question, p.expression, p.answer))
    # word_data: problem, expression, answer
    for p in WordProblemGenerator(seed=seed).generate(n=per_family):
        out.append(HRMExample("word", p.problem, p.expression, p.answer))
    # gsm_data: problem, expression, answer
    for p in GSMDataGenerator(seed=seed).generate(n=per_family):
        out.append(HRMExample("gsm", p.problem, p.expression, p.answer))
    # multi20_data: source, input, expression (no answer — compute via safe_eval)
    # Generate a multiple of 20 to ensure all sub-formats are represented.
    n_multi = max(per_family, 20) - (max(per_family, 20) % 20)
    for p in Multi20Generator(seed=seed).generate(n=n_multi):
        try:
            ans = safe_eval(p.expression)
            if isinstance(ans, float) and ans == int(ans):
                ans = int(ans)
            out.append(HRMExample("multi20", p.input, p.expression, str(ans)))
        except Exception:
            continue
    return out


def hrm_to_text(ex: HRMExample) -> tuple[str, int]:
    """Format an HRM example as `<|hrm|> question = expression<|eos|>`.
    Returns (text, asst_start_char)."""
    prefix = f"{HRM_MODE} {ex.question} = "
    text = f"{prefix}{ex.expression}{EOS_TOKEN}"
    return text, len(prefix)


# ----- Tokenizer -----

def train_bpe(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[UNK_TOKEN] + SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_with_mask(tok: Tokenizer, text: str, asst_start_char: int,
                     max_len: int) -> tuple[list[int], list[int]]:
    enc = tok.encode(text)
    ids = enc.ids[:max_len]
    offsets = enc.offsets[:max_len]
    mask = [1 if start >= asst_start_char else 0 for (start, _e) in offsets]
    return ids, mask


# ----- Training -----

def pad_batch(seqs, masks, modes, pad_id, max_len, dynamic_pad: bool = False):
    """Pad a batch of variable-length sequences.

    If dynamic_pad=True, pad only to the longest sequence in the batch (capped
    at max_len). For HRM-heavy training (where most examples are ~30 tokens
    in a max_len=384 config), this eliminates ~92% of pad-token compute on
    HRM batches while leaving LM batches at full length.
    """
    B = len(seqs)
    if dynamic_pad:
        batch_max = min(max(len(s) for s in seqs), max_len)
    else:
        batch_max = max_len
    input_ids = torch.full((B, batch_max), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, batch_max, dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = min(len(s), batch_max)
        input_ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        loss_mask[i, :L] = torch.tensor(m[:L], dtype=torch.float)
    return input_ids, loss_mask, modes


def train(model, tok, examples_with_mode, *,
          epochs, batch_size, lr, max_len,
          hrm_loss_weight: float = 4.0,
          hrm_iterations: int = 2,
          log_every: int = 50,
          device: str = "cpu",
          bf16: bool = False,
          length_bucket: bool = False):
    """Train. examples_with_mode: list of (text, asst_start, mode_name).

    bf16=True enables mixed-precision autocast on CUDA. 2-3× per-step throughput
    on Ampere+ GPUs (RTX 4070 is Ada = supports bf16 natively). No-op on CPU.
    """
    rng = random.Random(0)
    pad_id = tok.token_to_id(PAD_TOKEN)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = bf16 and device == "cuda"
    model.train()

    print(f"pre-tokenizing {len(examples_with_mode)} examples...", flush=True)
    cache = []
    for text, asst_start, mode in examples_with_mode:
        ids, mask = encode_with_mask(tok, text, asst_start, max_len)
        if sum(mask) > 0:
            cache.append((ids, mask, mode))
    print(f"  {len(cache)} after filter", flush=True)

    # Group by mode to enable per-mode iteration count.
    # For training simplicity, we still mix in batches but apply
    # iteration count = max over modes in batch (so we always use enough
    # iterations for HRM examples in the batch).
    steps_per_epoch = max(1, len(cache) // batch_size)
    t0 = time.time()
    global_step = 0
    for epoch in range(epochs):
        if length_bucket:
            # Sort by length, then shuffle within windows to preserve locality.
            cache.sort(key=lambda x: len(x[0]))
            window = batch_size * 4
            for w in range(0, len(cache), window):
                rng.shuffle(cache[w:w + window])
        else:
            rng.shuffle(cache)
        eloss, etoks = 0.0, 0
        for step in range(steps_per_epoch):
            batch = cache[step*batch_size:(step+1)*batch_size]
            if len(batch) < batch_size:
                continue
            seqs = [b[0] for b in batch]
            masks = [b[1] for b in batch]
            modes = [b[2] for b in batch]
            ids, lmask, modes = pad_batch(
                seqs, masks, modes, pad_id, max_len, dynamic_pad=length_bucket,
            )
            ids = ids.to(device)
            lmask = lmask.to(device)
            # Iteration count: max over batch (use HRM's higher count if any).
            n_iter = hrm_iterations if any(m == "hrm" for m in modes) else 1
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = model(ids[:, :-1], n_iterations=n_iter)
                targets = ids[:, 1:]
                mask = lmask[:, 1:]
                # Per-example mode weight: HRM weighted higher
                mode_weight = torch.tensor(
                    [hrm_loss_weight if m == "hrm" else 1.0 for m in modes],
                    dtype=torch.float, device=device,
                ).unsqueeze(1)                          # (B, 1)
                V = logits.size(-1)
                # Cast logits back to fp32 for CE numerical stability.
                per = F.cross_entropy(logits.reshape(-1, V).float(),
                                        targets.reshape(-1),
                                        reduction="none")
                per = per.reshape(targets.shape) * mask     # (B, L)
                weighted = per * mode_weight                # (B, L)
                # Loss = sum(weighted) / sum(mask * mode_weight)
                denom = (mask * mode_weight).sum().clamp(min=1.0)
                loss = weighted.sum() / denom
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += per.sum().item()                    # unweighted for ppl
            etoks += mask.sum().item()
            global_step += 1
            if global_step % log_every == 0:
                ppl = torch.tensor(eloss/max(1, etoks)).exp().item()
                print(f"  epoch {epoch+1}/{epochs} step {global_step} "
                      f"loss={eloss/max(1,etoks):.4f} ppl={ppl:.1f} "
                      f"{time.time()-t0:.1f}s", flush=True)
        ppl = torch.tensor(eloss/max(1, etoks)).exp().item()
        print(f"epoch {epoch+1}/{epochs} done  loss={eloss/max(1,etoks):.4f} "
              f"ppl={ppl:.1f}  {time.time()-t0:.1f}s", flush=True)
    model.eval()


# ----- Eval -----

@torch.no_grad()
def lm_perplexity(model, tok, examples, max_len, batch_size=8, device="cpu"):
    pad_id = tok.token_to_id(PAD_TOKEN)
    cache = []
    for text, asst_start, _mode in examples:
        ids, mask = encode_with_mask(tok, text, asst_start, max_len)
        if sum(mask) > 0:
            cache.append((ids, mask))
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(cache), batch_size):
        batch = cache[i:i+batch_size]
        seqs = [b[0] for b in batch]
        masks = [b[1] for b in batch]
        modes = ["lm"] * len(batch)
        ids, lmask, _ = pad_batch(seqs, masks, modes, pad_id, max_len)
        ids = ids.to(device); lmask = lmask.to(device)
        logits = model(ids[:, :-1], n_iterations=1)
        targets = ids[:, 1:]
        mask = lmask[:, 1:]
        V = logits.size(-1)
        per = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                reduction="none")
        masked = per * mask.reshape(-1)
        total_loss += masked.sum().item()
        total_tok += mask.sum().item()
    return torch.tensor(total_loss/max(1, total_tok)).exp().item()


@torch.no_grad()
def hrm_per_family_accuracy(model, tok, examples: list[HRMExample],
                             max_len, hrm_iterations: int, device: str = "cpu"):
    """Per-family accuracy: parse + interpret + value match."""
    eos_id = tok.token_to_id(EOS_TOKEN)
    by_family: dict[str, dict] = {}
    for ex in examples:
        prompt = f"{HRM_MODE} {ex.question} = "
        prompt_ids = tok.encode(prompt).ids
        ids = list(prompt_ids)
        new_tokens = []
        for _ in range(40):
            if len(ids) >= max_len:
                break
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x, n_iterations=hrm_iterations)[0, -1, :]
            nid = int(logits.argmax(-1).item())
            if nid == eos_id:
                break
            ids.append(nid)
            new_tokens.append(nid)
        gen_text = tok.decode(new_tokens).strip()
        bucket = by_family.setdefault(ex.family, {
            "total": 0, "parseable": 0, "structural": 0, "correct": 0,
        })
        bucket["total"] += 1
        try:
            g = parse_expression(gen_text.rstrip("=").strip())
            bucket["parseable"] += 1
            res = interpret(g)
            ans = str(res)
            if isinstance(res, float) and res == int(res):
                ans = str(int(res))
            if ans == ex.answer:
                bucket["correct"] += 1
            if gen_text.replace(" ", "") == ex.expression.replace(" ", ""):
                bucket["structural"] += 1
        except Exception:
            pass
    return by_family


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--per-family", type=int, default=600)
    ap.add_argument("--hrm-repeat", type=int, default=4)
    ap.add_argument("--hrm-loss-weight", type=float, default=4.0)
    ap.add_argument("--hrm-iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto",
                    help="cuda / cpu / auto (auto picks cuda if available)")
    ap.add_argument("--bf16", action="store_true",
                    help="Enable bfloat16 mixed-precision autocast on CUDA. "
                         "2-3× speedup on Ampere+ GPUs. No-op on CPU.")
    ap.add_argument("--compile", action="store_true",
                    help="Wrap model in torch.compile to reduce per-iteration "
                         "kernel launch overhead. Helps D5 recurrent substrate "
                         "(removes Python loop tax on GPU).")
    ap.add_argument("--length-bucket", action="store_true",
                    help="Sort examples by length + pad dynamically per batch. "
                         "For HRM-heavy training (short HRM + long LM examples), "
                         "cuts padding waste 3-4× on HRM batches.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Universal speedups for CUDA: TF32 for matmul, cudnn benchmark pickup.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"  device: {device}  bf16: {args.bf16 and device == 'cuda'}  "
          f"compile: {args.compile and device == 'cuda'}",
          flush=True)

    print("=== SubstrateHRMLM v2 — Main Thinking Engine ===")
    print(f"  d_model={args.d_model}  n_layers={args.n_layers}  "
          f"vocab={args.vocab_size}  max_len={args.max_len}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  HRM per-family={args.per_family}  repeat={args.hrm_repeat}x  "
          f"loss-weight={args.hrm_loss_weight}x  iterations={args.hrm_iterations}")

    # ---- Build corpus
    lm = load_lm_examples(LM_DATA_PATHS)
    print(f"\nloaded {len(lm)} LM examples", flush=True)

    hrm_pool = collect_hrm_examples(per_family=args.per_family, seed=42)
    print(f"collected {len(hrm_pool)} HRM examples across "
          f"{len({e.family for e in hrm_pool})} families", flush=True)

    # Per-family eval split: hold out last 30 of each family
    eval_per_family = 30
    families = sorted({e.family for e in hrm_pool})
    hrm_train: list[HRMExample] = []
    hrm_eval: list[HRMExample] = []
    for fam in families:
        fam_examples = [e for e in hrm_pool if e.family == fam]
        hrm_eval.extend(fam_examples[-eval_per_family:])
        hrm_train.extend(fam_examples[:-eval_per_family])
    print(f"  HRM train={len(hrm_train)}  HRM eval={len(hrm_eval)} "
          f"({eval_per_family} per family)", flush=True)

    # LM hold-out
    lm_train = lm[:-20]
    lm_eval = lm[-20:]

    # Build training set: HRM upsampled by repeat factor
    hrm_train_text = [(*hrm_to_text(e), "hrm") for e in hrm_train]
    hrm_train_repeated = hrm_train_text * args.hrm_repeat
    combined = lm_train + hrm_train_repeated
    random.Random(args.seed).shuffle(combined)
    n_lm = sum(1 for _t, _s, m in combined if m == "lm")
    n_hrm = sum(1 for _t, _s, m in combined if m == "hrm")
    print(f"\ncombined train={len(combined)} (LM={n_lm}, HRM={n_hrm})", flush=True)

    # ---- Tokenizer
    print("\ntraining BPE tokenizer...", flush=True)
    tok = train_bpe([t for t, _, _ in combined], vocab_size=args.vocab_size)
    print(f"  vocab: {tok.get_vocab_size()}", flush=True)

    # ---- Build combined-extension model
    layer_geometries = ["euclidean", "hyperbolic", "lattice", "euclidean"][:args.n_layers]
    while len(layer_geometries) < args.n_layers:
        layer_geometries.append("euclidean")
    print(f"  layer geometries: {layer_geometries}", flush=True)

    cfg = CombinedConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.d_model // 2,           # d_head=2 invariant
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_len=args.max_len,
        use_hard_max=False,
        layer_geometries=layer_geometries,
        default_iterations=1,
        max_iterations=4,
    )
    assert cfg.d_head == 2
    model = CombinedSmall2DTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)
    if args.compile and device == "cuda":
        # reduce-overhead mode uses CUDA graphs to amortize kernel launch cost
        # across D5 iteration steps — the main Python-loop tax.
        model = torch.compile(model, mode="reduce-overhead")
        print(f"  model compiled with torch.compile (reduce-overhead mode)",
              flush=True)

    # ---- Train
    print("\n--- training ---", flush=True)
    train(model, tok, combined,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          max_len=args.max_len,
          hrm_loss_weight=args.hrm_loss_weight,
          hrm_iterations=args.hrm_iterations,
          device=device,
          bf16=args.bf16,
          length_bucket=args.length_bucket)

    # ---- Save
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "substrate_hrmlm_v2.pt"
    tok_path = CKPT_DIR / "substrate_hrmlm_v2_tokenizer.json"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
    }, ckpt_path)
    tok.save(str(tok_path))
    print(f"\nsaved {ckpt_path.relative_to(REPO_ROOT)}", flush=True)

    # ---- Eval
    print("\n=== EVAL ===")
    print("\n[LM mode] perplexity on held-out claude_reasoning:")
    lm_ppl = lm_perplexity(model, tok, lm_eval, max_len=args.max_len,
                            device=device)
    print(f"  v2 LM ppl = {lm_ppl:.1f}  (v1 hybrid 386.9, "
          f"SubstrateLM MVP 424)", flush=True)

    print(f"\n[HRM mode] per-family correctness "
          f"(n_iterations={args.hrm_iterations}):")
    by_family = hrm_per_family_accuracy(
        model, tok, hrm_eval, max_len=args.max_len,
        hrm_iterations=args.hrm_iterations,
        device=device,
    )
    print(f"\n  family       parse    structural    correct")
    print(f"  ----------   ------   ----------    --------")
    best_correct_rate = 0.0
    for fam in sorted(by_family):
        b = by_family[fam]
        t = b["total"]
        cr = b["correct"] / t if t else 0
        sr = b["structural"] / t if t else 0
        pr = b["parseable"] / t if t else 0
        print(f"  {fam:<10}  {pr:>5.1%}    {sr:>5.1%}        {cr:>5.1%}")
        best_correct_rate = max(best_correct_rate, cr)

    # ---- Decision
    print("\n" + "=" * 60)
    print(f"Best HRM family correct rate: {best_correct_rate:.1%}")
    print(f"LM ppl: {lm_ppl:.1f}")
    pass_lm = lm_ppl <= 450
    if best_correct_rate >= 0.50 and pass_lm:
        print("DECISION: PASS — main thinking engine works at MVP scale.")
        print("  Architecture validated; scale-up to 100M+ for real engine.")
    elif best_correct_rate >= 0.20 and pass_lm:
        print(f"DECISION: PARTIAL — significant improvement over v1 (0% HRM)")
        print(f"  but didn't reach PASS threshold. Try more epochs or "
              f"capacity.")
    elif best_correct_rate >= 0.05:
        print(f"DECISION: WEAK SIGNAL — HRM mode learning but slowly.")
        print(f"  Likely needs more epochs or hrm-loss-weight increased.")
    else:
        print(f"DECISION: FAIL — HRM mode still not learning structure.")
        print(f"  Re-examine mode-token signal strength or geometry choices.")
    print("=" * 60)


if __name__ == "__main__":
    main()
