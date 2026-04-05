"""Stage 1: Train a reasoning base model from qwen3:0.6b + Claude reasoning data.

This creates a 'smarter' 0.6B base that all specialists inherit from.

Usage:
    python -m agents.distill.train_base
"""

import json
import sys
from pathlib import Path

from agents.distill.config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    MERGED_DIR,
    QLORA_CONFIG,
    STUDENT_BASE,
)

REASONING_DATA = DATA_DIR / "claude_reasoning.jsonl"
CODING_REASONING_DATA = DATA_DIR / "coding_reasoning_claude.jsonl"
REASONING_CHECKPOINT = CHECKPOINTS_DIR / "reasoning_base"
REASONING_MERGED = MERGED_DIR / "reasoning_base"


def check_dependencies():
    missing = []
    for pkg in ["torch", "unsloth", "transformers", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("pip install " + " ".join(missing))
        sys.exit(1)


def train_reasoning_base():
    """Train the reasoning base model."""
    check_dependencies()

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    if not REASONING_DATA.exists():
        print(f"Error: {REASONING_DATA} not found. Run fetch_datasets.py first.")
        return

    REASONING_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    REASONING_MERGED.mkdir(parents=True, exist_ok=True)

    # Load data (claude_reasoning.jsonl contains filtered HF + hand-written after merge)
    examples = []
    with open(REASONING_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"Stage 1: Training Reasoning Base")
    print(f"Base model: Qwen/Qwen3.5-0.8B")
    print(f"Training examples: {len(examples)}")
    print(f"{'='*60}\n")

    # Load model — using Qwen 3.5 0.8B (newer architecture)
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-0.8B",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply QLoRA
    print("Applying QLoRA adapters...")
    cfg = QLORA_CONFIG
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Format dataset
    print("Formatting dataset...")
    dataset = Dataset.from_list(examples)

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    # Training — single epoch like the TeichAI approach
    # Batch=1 to fit in 8GB VRAM (Qwen3.5 has 248K vocab = huge CE loss)
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03,
        num_train_epochs=3,  # 3 epochs on curated data — diverse enough to avoid memorization
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir=str(REASONING_CHECKPOINT),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",
    )

    print("Starting training (1 epoch, train_on_responses_only)...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,  # Disable packing to save VRAM
    )

    # Mask instruction tokens — only compute loss on model responses
    from unsloth import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    # Merge
    print("\nMerging LoRA adapters...")
    model.save_pretrained_merged(
        str(REASONING_MERGED),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Reasoning base saved to: {REASONING_MERGED}")
    print(f"\nThis model is now the base for all specialist training.")
    print(f"Update STUDENT_BASE in config.py to point to: {REASONING_MERGED}")

    return REASONING_MERGED


if __name__ == "__main__":
    train_reasoning_base()
