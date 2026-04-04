"""QLoRA fine-tuning script for specialist models. Run in WSL2 with GPU access.

Usage:
    python -m agents.distill.train --domain python
    python -m agents.distill.train --domain all
"""

import argparse
import json
import sys
from pathlib import Path

from agents.distill.config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DOMAINS,
    MERGED_DIR,
    QLORA_CONFIG,
    STUDENT_BASE,
)
from agents.distill.train_base import REASONING_MERGED


def check_dependencies():
    """Verify that training dependencies are installed."""
    missing = []
    for pkg in ["torch", "unsloth", "transformers", "trl", "datasets", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def load_dataset(data_path: Path, max_seq_length: int):
    """Load JSONL training data and format for SFTTrainer."""
    from datasets import Dataset

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"  Loaded {len(examples)} examples from {data_path}")
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Format a chat example using the model's chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train_specialist(domain: str):
    """Fine-tune a QLoRA adapter on domain-specific data."""
    check_dependencies()

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    if domain not in DOMAINS:
        print(f"Error: Unknown domain '{domain}'. Available: {list(DOMAINS.keys())}")
        return

    data_path = DATA_DIR / f"{domain}.jsonl"
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        print(f"Run: python -m agents.distill.generate --domain {domain}")
        return

    checkpoint_dir = CHECKPOINTS_DIR / domain
    merged_dir = MERGED_DIR / domain
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    cfg = QLORA_CONFIG

    # Use reasoning base if available, otherwise vanilla 0.6B
    base_model = str(REASONING_MERGED) if REASONING_MERGED.exists() else STUDENT_BASE

    print(f"\n{'='*60}")
    print(f"Training specialist: {domain}")
    print(f"Base model: {base_model} {'(reasoning base)' if REASONING_MERGED.exists() else '(vanilla)'}")
    print(f"Data: {data_path}")
    print(f"LoRA rank: {cfg['r']}, alpha: {cfg['lora_alpha']}")
    print(f"Epochs: {cfg['num_train_epochs']}, batch: {cfg['per_device_train_batch_size']}")
    print(f"{'='*60}\n")

    # Load base model with 4-bit quantization
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=cfg["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply QLoRA adapters
    print("Applying QLoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and format dataset
    print("Loading dataset...")
    dataset = load_dataset(data_path, cfg["max_seq_length"])
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir=str(checkpoint_dir),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=True,
    )

    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    # Merge LoRA adapters into base model
    print("\nMerging LoRA adapters...")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {merged_dir}")

    return merged_dir


def main():
    parser = argparse.ArgumentParser(description="Train specialist models with QLoRA")
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help=f"Domain to train. Options: {', '.join(DOMAINS.keys())}, all",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    args = parser.parse_args()

    if args.epochs:
        QLORA_CONFIG["num_train_epochs"] = args.epochs

    if args.domain == "all":
        for domain in DOMAINS:
            train_specialist(domain)
    else:
        train_specialist(args.domain)


if __name__ == "__main__":
    main()
