"""
HRM Training — train domain-specific HRM models on CALM-generated data.

Usage:
    python3 -m calm.hrm.train --domain math --epochs 1000

    from calm.hrm.train import HRMTrainer
    trainer = HRMTrainer()
    trainer.train(epochs=1000)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from calm.hrm.model import HRM, HRMSeq2Seq, HRMConfig
from calm.hrm.data import MathDataGenerator, MathDataset, MathSeq2SeqDataset, VOCAB_SIZE


DEFAULT_CHECKPOINT_DIR = Path("calm/hrm/checkpoints")


class HRMTrainer:
    """Train an HRM model on generated math data."""

    def __init__(
        self,
        config: Optional[HRMConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    ):
        self.config = config or HRMConfig(vocab_size=VOCAB_SIZE)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        num_problems: int = 1000,
        epochs: int = 1000,
        batch_size: int = 64,
        lr: float = 1e-3,
        eval_interval: int = 100,
        verbose: bool = True,
    ) -> HRM:
        """Train an HRM model on generated math data.

        Returns the trained model.
        """
        # Generate data
        if verbose:
            print(f"[hrm] Generating {num_problems} math problems...")
        gen = MathDataGenerator()
        problems = gen.generate(num_problems)
        if verbose:
            print(f"[hrm] Generated {len(problems)} problems (difficulties 1-5)")

        # Split train/val
        dataset = MathDataset(problems, max_len=self.config.max_seq_len)
        val_size = max(1, len(dataset) // 10)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Create model
        model = HRM(self.config).to(self.device)
        if verbose:
            print(f"[hrm] Model: {model.param_count():,} params on {self.device}")

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        best_val_acc = 0.0
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                logits = model(input_ids)

                # Loss only on answer region (where loss_mask is True)
                logits_flat = logits.reshape(-1, self.config.vocab_size)
                targets_flat = target_ids.reshape(-1)
                mask_flat = loss_mask.reshape(-1)

                if mask_flat.any():
                    loss = F.cross_entropy(
                        logits_flat[mask_flat],
                        targets_flat[mask_flat],
                        ignore_index=0,
                    )
                else:
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)

            # Evaluate
            if epoch % eval_interval == 0 or epoch == 1:
                val_acc = self._evaluate(model, val_loader)
                elapsed = time.time() - t_start

                if verbose:
                    print(f"[hrm] epoch {epoch:4d}/{epochs}: "
                          f"loss={avg_loss:.4f}, val_acc={val_acc:.1%}, "
                          f"lr={scheduler.get_last_lr()[0]:.6f}, "
                          f"elapsed={elapsed:.0f}s")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(model, epoch, val_acc)

        if verbose:
            total_time = time.time() - t_start
            print(f"[hrm] Training complete: {total_time:.0f}s, "
                  f"best val_acc={best_val_acc:.1%}")

        return model

    def _evaluate(self, model: HRM, val_loader: DataLoader) -> float:
        """Evaluate model accuracy on validation set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                logits = model(input_ids)
                preds = logits.argmax(-1)

                # Check only answer region (where loss_mask is True)
                mask = loss_mask & (target_ids != 0)
                correct += (preds[mask] == target_ids[mask]).sum().item()
                total += mask.sum().item()

        return correct / max(total, 1)

    def _save_checkpoint(self, model: HRM, epoch: int, val_acc: float):
        """Save model checkpoint."""
        path = self.checkpoint_dir / "math_hrm_best.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_heads": self.config.num_heads,
                "expansion": self.config.expansion,
                "L_layers": self.config.L_layers,
                "H_layers": self.config.H_layers,
                "L_cycles": self.config.L_cycles,
                "H_cycles": self.config.H_cycles,
                "max_seq_len": self.config.max_seq_len,
            },
            "epoch": epoch,
            "val_acc": val_acc,
        }, path)


class HRMSeq2SeqTrainer:
    """Train an HRMSeq2Seq model (encoder-decoder, optionally scratchpad)."""

    def __init__(
        self,
        config: Optional[HRMConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
        reverse_digits: bool = True,
        scratchpad: bool = False,
        structure_only: bool = False,
    ):
        self.config = config or HRMConfig(vocab_size=VOCAB_SIZE)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scratchpad = scratchpad
        self.structure_only = structure_only
        # Scratchpad traces mix intermediate numbers with <call> tokens; reversing
        # digits would corrupt them. Force off when scratchpad is on.
        # Structure-only also disables reverse_digits (target is the problem expression).
        self.reverse_digits = reverse_digits and not scratchpad and not structure_only

    def train(
        self,
        num_problems: int = 2000,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        eval_interval: int = 50,
        verbose: bool = True,
    ) -> HRMSeq2Seq:
        if verbose:
            if self.structure_only:
                mode = "structure-only"
            elif self.scratchpad:
                mode = "scratchpad"
            else:
                mode = "answer-only"
            print(f"[hrm-s2s] Generating {num_problems} math problems "
                  f"(mode={mode}, reverse_digits={self.reverse_digits})...")
        gen = MathDataGenerator()
        # Structure-only target doesn't need the scratchpad trace payload.
        want_trace = self.scratchpad and not self.structure_only
        problems = gen.generate(num_problems, trace=want_trace)
        if verbose:
            print(f"[hrm-s2s] Generated {len(problems)} problems (difficulties 1-5)")

        max_enc = self.config.max_seq_len
        max_dec = self.config.max_dec_len
        dataset = MathSeq2SeqDataset(problems, max_enc_len=max_enc,
                                      max_dec_len=max_dec,
                                      reverse_digits=self.reverse_digits,
                                      use_trace=want_trace,
                                      structure_only=self.structure_only)
        val_size = max(1, len(dataset) // 10)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = HRMSeq2Seq(self.config).to(self.device)
        if verbose:
            enc_p = sum(p.numel() for p in model.encoder.parameters())
            dec_p = sum(p.numel() for p in model.decoder.parameters())
            print(f"[hrm-s2s] Model: {model.param_count():,} params "
                  f"(enc {enc_p:,} + dec {dec_p:,}) on {self.device}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                enc = batch["encoder_ids"].to(self.device)
                dec_in = batch["decoder_input_ids"].to(self.device)
                dec_tgt = batch["decoder_target_ids"].to(self.device)
                mask = batch["loss_mask"].to(self.device)

                logits = model(enc, dec_in)
                logits_flat = logits.reshape(-1, self.config.vocab_size)
                targets_flat = dec_tgt.reshape(-1)
                mask_flat = mask.reshape(-1)
                if not mask_flat.any():
                    continue
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)

            if epoch % eval_interval == 0 or epoch == 1:
                val_acc = self._evaluate(model, val_loader)
                elapsed = time.time() - t_start
                if verbose:
                    print(f"[hrm-s2s] epoch {epoch:4d}/{epochs}: "
                          f"loss={avg_loss:.4f}, val_acc={val_acc:.1%}, "
                          f"lr={scheduler.get_last_lr()[0]:.6f}, "
                          f"elapsed={elapsed:.0f}s")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(model, epoch, val_acc)

        if verbose:
            total_time = time.time() - t_start
            print(f"[hrm-s2s] Training complete: {total_time:.0f}s, best val_acc={best_val_acc:.1%}")
        return model

    def _evaluate(self, model: HRMSeq2Seq, val_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                enc = batch["encoder_ids"].to(self.device)
                dec_in = batch["decoder_input_ids"].to(self.device)
                dec_tgt = batch["decoder_target_ids"].to(self.device)
                mask = batch["loss_mask"].to(self.device)
                logits = model(enc, dec_in)
                preds = logits.argmax(-1)
                correct += (preds[mask] == dec_tgt[mask]).sum().item()
                total += mask.sum().item()
        return correct / max(total, 1)

    def _save_checkpoint(self, model: HRMSeq2Seq, epoch: int, val_acc: float):
        if self.structure_only:
            fname = "math_structure_best.pt"
        elif self.scratchpad:
            fname = "math_scratchpad_best.pt"
        elif not self.reverse_digits:
            fname = "math_seq2seq_best_no_reverse.pt"
        else:
            fname = "math_seq2seq_best.pt"
        path = self.checkpoint_dir / fname
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_heads": self.config.num_heads,
                "expansion": self.config.expansion,
                "L_layers": self.config.L_layers,
                "H_layers": self.config.H_layers,
                "L_cycles": self.config.L_cycles,
                "H_cycles": self.config.H_cycles,
                "max_seq_len": self.config.max_seq_len,
                "decoder_layers": self.config.decoder_layers,
                "max_dec_len": self.config.max_dec_len,
            },
            "epoch": epoch,
            "val_acc": val_acc,
            "reverse_digits": self.reverse_digits,
            "scratchpad": self.scratchpad,
            "structure_only": self.structure_only,
        }, path)


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Train HRM math model")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--problems", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq2seq", action="store_true",
                        help="Use HRMSeq2Seq (encoder-decoder) instead of legacy HRM.")
    parser.add_argument("--scratchpad", action="store_true",
                        help="Seq2seq only: train on step-by-step traces with <call>/<end_call> delegation.")
    parser.add_argument("--structure-only", action="store_true",
                        help="Seq2seq only: target is `problem =` (echo + terminator). Values come from "
                             "the LLM-Computer interpreter at inference time via --verified.")
    parser.add_argument("--l-layers", type=int, default=1, help="Seq2seq only: encoder L_layers count.")
    parser.add_argument("--h-layers", type=int, default=1, help="Seq2seq only: encoder H_layers count.")
    parser.add_argument("--num-heads", type=int, default=4, help="Seq2seq only: attention heads per block.")
    parser.add_argument("--no-reverse-digits", dest="reverse_digits", action="store_false",
                        default=True, help="Seq2seq only: disable digit-reversal for ablation.")
    parser.add_argument("--dec-layers", type=int, default=2, help="Seq2seq only: decoder block count.")
    parser.add_argument("--max-enc", type=int, default=32, help="Seq2seq only: max encoder length.")
    parser.add_argument("--max-dec", type=int, default=16, help="Seq2seq only: max decoder length.")
    args = parser.parse_args()

    if args.seq2seq:
        config = HRMConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=args.hidden,
            num_heads=args.num_heads,
            L_layers=args.l_layers,
            H_layers=args.h_layers,
            max_seq_len=args.max_enc,
            max_dec_len=args.max_dec,
            decoder_layers=args.dec_layers,
        )
        trainer = HRMSeq2SeqTrainer(config=config, device=args.device,
                                     reverse_digits=args.reverse_digits,
                                     scratchpad=args.scratchpad,
                                     structure_only=args.structure_only)
        model = trainer.train(
            num_problems=args.problems,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        config = HRMConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=args.hidden,
            max_seq_len=64,
        )
        trainer = HRMTrainer(config=config, device=args.device)
        model = trainer.train(
            num_problems=args.problems,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    print(f"Final param count: {model.param_count():,}")


if __name__ == "__main__":
    main()
