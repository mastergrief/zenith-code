"""
Cloud training script for Qwen 3.5 4B reasoning base.
Run on RunPod/Colab with an A100 or A40 GPU.

Usage:
    1. Upload this file + data/ directory to the pod
    2. pip install unsloth
    3. python train_4b_cloud.py
    4. Download the merged model from ./merged_4b/
"""

import json
import sys
from pathlib import Path

# --- Config ---
MODEL_NAME = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 1024  # Qwen 3.5 248K vocab needs conservative seq_len
LORA_R = 16
LORA_ALPHA = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1  # 248K vocab CE loss is huge — start at 1, 24GB should handle it
GRAD_ACCUM = 16  # effective batch = 1 * 16 = 16

DATA_FILE = Path("data/claude_reasoning.jsonl")  # merged training file (HF + hand-written)
OUTPUT_DIR = Path("./checkpoints_4b")
MERGED_DIR = Path("./merged_4b")


def main():
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found.")
        print("Upload your data files first:")
        print("  - data/claude_reasoning.jsonl (merged training file)")
        sys.exit(1)

    # Load data
    examples = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"Training Qwen 3.5 4B Reasoning Base (Cloud)")
    print(f"Model: {MODEL_NAME}")
    print(f"Examples: {len(examples)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Seq length: {MAX_SEQ_LENGTH}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Apply QLoRA
    print("Applying QLoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
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

    # Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.03,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        output_dir=str(OUTPUT_DIR),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",
    )

    print(f"Starting training ({NUM_EPOCHS} epochs, train_on_responses_only)...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # Mask instruction tokens
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
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    # Merge and save
    print("\nMerging LoRA adapters (16-bit)...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {MERGED_DIR}")

    # Also save as GGUF Q5_K_M directly if llama.cpp is available
    try:
        print("\nExporting to GGUF Q5_K_M...")
        model.save_pretrained_gguf(
            "merged_4b_gguf",
            tokenizer,
            quantization_method="q5_k_m",
        )
        print("GGUF saved to: merged_4b_gguf/")
    except Exception as e:
        print(f"GGUF export failed (not critical): {e}")
        print("You can convert locally with llama.cpp instead.")

    print("\n" + "="*60)
    print("DONE! Download these files:")
    print(f"  {MERGED_DIR}/  (FP16 safetensors, ~9GB)")
    print(f"  merged_4b_gguf/  (Q5_K_M GGUF, ~4GB) — if export succeeded")
    print("="*60)


if __name__ == "__main__":
    main()
