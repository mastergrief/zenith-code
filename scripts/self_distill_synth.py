"""Self-distillation: retrain synth-A on library-augmented data.

Pipeline:
  1. Load the current synth-A checkpoint + library.jsonl.
  2. Build an augmented dataset: original Family A templates PLUS every
     expression currently in the library (taught via !correct or
     discovered earlier).
  3. Fine-tune synth-A from its existing checkpoint on the augmented
     distribution for a modest number of epochs.
  4. Evaluate on held-out tasks, including BOTH original and
     library-contributed templates. Report functional-correctness
     pre- vs post-fine-tune.

If the library has a / 2 (from the !correct teach), post-fine-tune
synth-A should autonomously discover halve tasks without human help.
This closes the loop: teach once (1 line), the model absorbs it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE
from calm.hrm.model import HRMConfig, HRMSeq2Seq
from calm.llm_computer.synth.data import SynthFamilyADataset
from calm.llm_computer.synth.data_augmented import AugmentedSynthGenerator
from calm.llm_computer.synth.infer import SynthFamilyAReasoner, functional_correct
from calm.llm_computer.synth.library import DEFAULT_LIBRARY_PATH, Library


SOURCE_CKPT = Path("calm/hrm/checkpoints/synth_familyA_best.pt")
DISTILLED_CKPT = Path("calm/hrm/checkpoints/synth_familyA_distilled.pt")


def _evaluate_checkpoint(ckpt_path: Path, tasks):
    """Return (exact_count, functional_count, total) for a saved checkpoint."""
    reasoner = SynthFamilyAReasoner(str(ckpt_path))
    exact = 0
    functional = 0
    for sample in tasks:
        emit = reasoner.predict(sample)
        if emit.replace(" ", "") == sample.template.replace(" ", ""):
            exact += 1
        if functional_correct(emit, sample):
            functional += 1
    return exact, functional, len(tasks)


def _build_dataset(extras, n_problems: int, seed: int,
                    max_enc: int, max_dec: int):
    gen = AugmentedSynthGenerator(seed=seed, extra_templates=extras)
    samples = gen.generate(n_problems)
    return SynthFamilyADataset(samples, max_enc_len=max_enc, max_dec_len=max_dec)


def fine_tune(extras, epochs: int, n_problems: int,
              batch_size: int, lr: float, max_enc: int, max_dec: int,
              seed: int = 12345):
    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(SOURCE_CKPT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(SOURCE_CKPT, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = HRMConfig(**cfg_dict)
    model = HRMSeq2Seq(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[self-distill] warmstarted from {SOURCE_CKPT.name} "
          f"(epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', 0):.1%})")
    print(f"[self-distill] {model.param_count():,} params on {device}")

    ds = _build_dataset(extras, n_problems, seed, max_enc, max_dec)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Fine-tune with lower LR to avoid catastrophic forgetting.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; nb = 0
        for b in train_loader:
            enc = b["encoder_ids"].to(device)
            din = b["decoder_input_ids"].to(device)
            dt = b["decoder_target_ids"].to(device)
            m = b["loss_mask"].to(device)
            logits = model(enc, din)
            lf = logits.reshape(-1, VOCAB_SIZE)
            tf = dt.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = F.cross_entropy(lf[mf], tf[mf])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct, tot = 0, 0
            with torch.no_grad():
                for b in val_loader:
                    enc = b["encoder_ids"].to(device)
                    din = b["decoder_input_ids"].to(device)
                    dt = b["decoder_target_ids"].to(device)
                    m = b["loss_mask"].to(device)
                    preds = model(enc, din).argmax(-1)
                    correct += (preds[m] == dt[m]).sum().item()
                    tot += m.sum().item()
            va = correct / max(tot, 1)
            print(f"[self-distill] epoch {epoch:3d}/{epochs}: "
                  f"loss={total/max(nb,1):.4f}, val_acc={va:.1%}, "
                  f"elapsed={time.time()-t0:.0f}s")
            if va > best:
                best = va
                DISTILLED_CKPT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg_dict,
                    "epoch": ckpt.get("epoch", 0) + epoch,
                    "val_acc": va,
                    "self_distilled": True,
                    "extras_used": extras,
                }, DISTILLED_CKPT)
    print(f"[self-distill] done in {time.time()-t0:.0f}s, best val_acc={best:.1%}")
    return DISTILLED_CKPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--problems", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-enc", type=int, default=96)
    p.add_argument("--max-dec", type=int, default=16)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--library", default=str(DEFAULT_LIBRARY_PATH))
    args = p.parse_args()

    # Load library for extras.
    library = Library(path=Path(args.library))
    extras = [e.expression for e in library]
    print(f"[self-distill] library at {args.library}: "
          f"{len(extras)} entries — {extras}")
    if not extras:
        print("[self-distill] library is empty — nothing to distill")
        return

    # Generate held-out eval tasks covering BOTH original and extras.
    gen = AugmentedSynthGenerator(seed=9999, extra_templates=extras)
    eval_tasks = gen.generate(args.eval_n)

    # Pre-fine-tune baseline.
    print(f"\n[self-distill] PRE-FINE-TUNE eval ({args.eval_n} tasks)")
    pre_exact, pre_func, total = _evaluate_checkpoint(SOURCE_CKPT, eval_tasks)
    print(f"  exact:      {pre_exact}/{total} = {pre_exact/total:.0%}")
    print(f"  functional: {pre_func}/{total} = {pre_func/total:.0%}")
    # Per-extra breakdown
    for t in extras:
        matching = [s for s in eval_tasks if s.template == t]
        if matching:
            ok = sum(
                1 for s in matching
                if functional_correct(
                    SynthFamilyAReasoner(str(SOURCE_CKPT)).predict(s), s)
            )
            print(f"    [extra: {t!r}] pre: {ok}/{len(matching)}")

    # Fine-tune
    print(f"\n[self-distill] fine-tuning for {args.epochs} epochs, "
          f"{args.problems} problems, lr={args.lr}")
    distilled_ckpt = fine_tune(
        extras=extras, epochs=args.epochs, n_problems=args.problems,
        batch_size=args.batch_size, lr=args.lr,
        max_enc=args.max_enc, max_dec=args.max_dec,
    )

    # Post-fine-tune eval
    print(f"\n[self-distill] POST-FINE-TUNE eval ({args.eval_n} tasks)")
    post_exact, post_func, total = _evaluate_checkpoint(distilled_ckpt, eval_tasks)
    print(f"  exact:      {post_exact}/{total} = {post_exact/total:.0%}  "
          f"(Δ {post_exact-pre_exact:+d})")
    print(f"  functional: {post_func}/{total} = {post_func/total:.0%}  "
          f"(Δ {post_func-pre_func:+d})")
    for t in extras:
        matching = [s for s in eval_tasks if s.template == t]
        if matching:
            reasoner = SynthFamilyAReasoner(str(distilled_ckpt))
            ok = sum(
                1 for s in matching
                if functional_correct(reasoner.predict(s), s)
            )
            print(f"    [extra: {t!r}] post: {ok}/{len(matching)}")


if __name__ == "__main__":
    main()
