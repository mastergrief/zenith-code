"""Train copy-augmented SubstrateHRM — pointer-copy transducer.

Same task as train_substrate_hrm.py (NL → expression), but the model
has a learned copy gate + pointer attention. Digits get copied from
the input prefix; operators get generated from vocabulary.

Key differences from base training:
- Model returns log-probs (not raw logits) — use NLL loss
- Scheduled sampling still applies (hardens autoregressive decode)
- Checkpoint saved on best autoreg accuracy (same gate metric)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.data import VOCAB_SIZE, _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.copy_augmented import build_copy_augmented_hrm
from scripts.train_substrate_hrm import SeqDataset


CHECKPOINT_PATH = Path("calm/hrm/checkpoints/copy_augmented_hrm_best.pt")


def _autoreg_eval(model, problems, device, max_gen=30):
    """Autoregressive eval — greedy decode, check exact expression match."""
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    eos = _CHAR_TO_ID["<eos>"]
    model.eval()
    correct = 0
    for p in problems:
        ids = [bos] + [_CHAR_TO_ID[c] for c in p.question
                       if c in _CHAR_TO_ID] + [sep]
        gen = []
        for _ in range(max_gen):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                log_probs = model(x)
            nxt = int(log_probs[0, -1].argmax().item())
            if nxt == eos:
                break
            gen.append(nxt)
            ids.append(nxt)
        decoded = "".join(
            _ID_TO_CHAR.get(i, "") for i in gen
            if not _ID_TO_CHAR.get(i, "").startswith("<")
        ).strip().rstrip("=").strip()
        expected = p.expression.strip()
        if decoded == expected:
            correct += 1
    return correct / max(len(problems), 1)


def _evaluate(model, loader, device):
    """Teacher-forced accuracy on masked positions."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            log_probs = model(x)
            preds = log_probs.argmax(-1)
            correct += (preds[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def _copy_gate_stats(model, loader, device):
    """Report average copy gate activation on prefix vs expression positions."""
    model.eval()
    prefix_gates, expr_gates = [], []
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(device)
            m = b["loss_mask"].to(device)
            B, S = x.shape

            # Get hidden states after transformer layers
            pos_idx = torch.arange(S, device=device)
            h = model.tok(x) + model.pos(pos_idx)
            for layer in range(model.config.n_layers):
                qkv = model.W_qkv[layer](h)
                qkv = qkv.reshape(B, S, 3, model.config.n_heads, model.config.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn = model._attention(q, k, v, hard_max=model.config.use_hard_max)
                attn = attn.transpose(1, 2).reshape(B, S, model.config.d_model)
                h = h + model.W_out[layer](attn)
                gate, val = model.ff_in[layer](h).chunk(2, dim=-1)
                h = h + model.ff_out[layer](F.relu(gate) * val)

            p_copy = torch.sigmoid(model.copy_gate(h)).squeeze(-1)  # (B, S)
            prefix_gates.append(p_copy[~m].mean().item())
            expr_gates.append(p_copy[m].mean().item())
            if len(prefix_gates) >= 5:
                break
    return (sum(prefix_gates) / len(prefix_gates),
            sum(expr_gates) / len(expr_gates))


def train(epochs=500, problems=5000, batch_size=64, lr=1e-3,
          d_model=64, n_heads=32, n_layers=4, d_ffn=128,
          max_len=96, n_copy_heads=4, seed=42, eval_every=10,
          scheduled_sampling=True, tf_ratio_start=1.0, tf_ratio_end=0.3,
          device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = NLMathDataGenerator(seed=seed)
    probs = gen.generate(problems)
    ds = SeqDataset(probs, max_len=max_len)
    val_size = max(1, len(ds) // 10)
    train_set, val_set = random_split(ds, [len(ds) - val_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    val_probs = [probs[i] for i in val_set.indices]

    model = build_copy_augmented_hrm(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, use_hard_max=False,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    copy_params = (sum(p.numel() for p in model.copy_gate.parameters()) +
                   sum(p.numel() for p in model.copy_q_proj.parameters()) +
                   sum(p.numel() for p in model.copy_k_proj.parameters()))
    print(f"[copy-hrm] model: {total_params:,} params ({copy_params:,} copy mechanism) on {device}")
    if scheduled_sampling:
        print(f"[copy-hrm] scheduled sampling: tf_ratio {tf_ratio_start:.1f}"
              f" → {tf_ratio_end:.1f} over {epochs} epochs")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_autoreg = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * (epoch / epochs)

        for b in train_loader:
            x = b["input_ids"].to(device)
            y = b["target_ids"].to(device)
            m = b["loss_mask"].to(device)
            B, S = x.shape

            if scheduled_sampling and tf_ratio < 0.99:
                with torch.no_grad():
                    pred_log_probs = model(x)
                    preds = pred_log_probs.argmax(-1)
                swap = (torch.rand(B, S, device=device) > tf_ratio) & m
                swap_shifted = torch.zeros_like(swap)
                swap_shifted[:, 1:] = swap[:, :-1]
                preds_shifted = torch.zeros_like(x)
                preds_shifted[:, 1:] = preds[:, :-1]
                modified_x = torch.where(swap_shifted, preds_shifted, x)
                log_probs = model(modified_x)
            else:
                log_probs = model(x)

            # NLL loss on masked positions (model outputs log-probs)
            lp = log_probs.reshape(-1, VOCAB_SIZE)
            tf = y.reshape(-1)
            mf = m.reshape(-1)
            if not mf.any():
                continue
            loss = F.nll_loss(lp[mf], tf[mf])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1

        sched.step()

        if epoch % eval_every == 0 or epoch == 1:
            va = _evaluate(model, val_loader, device)
            autoreg = _autoreg_eval(model, val_probs[:50], device)
            extra = ""
            if epoch % 50 == 0:
                pg, eg = _copy_gate_stats(model, val_loader, device)
                extra = f", gate_prefix={pg:.2f}, gate_expr={eg:.2f}"
            print(f"[copy-hrm] epoch {epoch:4d}/{epochs}: "
                  f"loss={total_loss/max(nb,1):.4f}, val_acc={va:.1%}, "
                  f"autoreg={autoreg:.1%}, tf_ratio={tf_ratio:.2f}, "
                  f"elapsed={time.time()-t0:.0f}s{extra}")
            if autoreg > best_autoreg:
                best_autoreg = autoreg
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "d_model": d_model,
                        "n_heads": n_heads, "n_layers": n_layers,
                        "d_ffn": d_ffn, "max_len": max_len,
                        "n_copy_heads": n_copy_heads,
                    },
                    "epoch": epoch, "val_acc": va,
                    "autoreg_acc": autoreg, "copy_augmented": True,
                }, CHECKPOINT_PATH)

    print(f"[copy-hrm] done: {time.time()-t0:.0f}s, best autoreg={best_autoreg:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--problems", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=128)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--n-copy-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--no-scheduled-sampling", action="store_true")
    p.add_argument("--tf-ratio-end", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    train(
        epochs=args.epochs, problems=args.problems,
        batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len, n_copy_heads=args.n_copy_heads,
        seed=args.seed, eval_every=args.eval_every,
        scheduled_sampling=not args.no_scheduled_sampling,
        tf_ratio_end=args.tf_ratio_end, device=args.device,
    )


if __name__ == "__main__":
    main()
