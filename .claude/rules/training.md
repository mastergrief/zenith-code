# Training Rules

## VRAM Budget (8 GB RTX 4070 Laptop)
- Qwen 3 0.6B QLoRA: ~2.5 GB — fits with batch=4, packing=true
- Qwen 3.5 0.8B QLoRA: ~4 GB — requires batch=1, packing=false, seq_len=1024
- Qwen 3.5 4B: too large for QLoRA on 8 GB
- Always stop Ollama before training: `ollama stop <model>` or verify `ollama ps` is empty

## Priority Order
- **Data quality > data quantity > model size > training tricks.** One hour writing 20 high-quality examples beats hours of hyperparameter tuning.
- Each example should demonstrate the *reasoning process* (`<think>` block), not just the answer
- Match the training domain to the task: coding data for coding models, routing data for routing models

## Training Best Practices
- Always use `train_on_responses_only` — masks instruction/prompt tokens so loss is only computed on the model's generated responses, not on system prompts or user messages. Also slightly faster.
- ~4K examples is the sweet spot for reasoning distillation (confirmed by TeichAI, Jackrong, and our own runs)
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` is the best-filtered Claude reasoning dataset on HuggingFace
- Single epoch prevents catastrophic forgetting — more epochs on small datasets = memorization, not generalization
- Filter training data before use: `python -m agents.distill.filter_reasoning --merge`
- Filter aggressively — removing bad data improves results more than adding mediocre data

## Dataset Quality
- Claude-authored data >> 9B-generated data (higher quality, more consistent, fewer examples needed)
- HuggingFace datasets (nohurry, TeichAI, Crownelius) provide Claude Opus reasoning traces
- Hand-written data committed to repo: `coding_reasoning_claude.jsonl` (90 examples), `orchestrator_claude.jsonl` (121 examples)
- Filter out: hallucinated facts (Bitcoin prices, sports results), non-technical content (opinion debates, riddles), junk (<200 char responses)
- Orchestrator domain is classification (route to specialist), not code generation
- Specialist datasets are still small (25-53 examples) — need expansion

## Known Issues
- Qwen 3.5 has 248K vocab which makes fused cross-entropy loss OOM on 8 GB
  - Fix: batch_size=1, max_seq_length=1024, packing=False
- Unsloth compiled cache stored at `unsloth_compiled_cache/` in project root (gitignored)
- Git Bash mangles WSL paths with parentheses in PATH — use `wsl -e bash -c` or write scripts to `/tmp/`
- No Thunderbolt/eGPU on Acer Nitro AN17-42 — cloud GPUs are the path to training bigger models

## Export
- GGUF conversion requires llama.cpp (`git clone https://github.com/ggml-org/llama.cpp`)
- Quantize to Q4_K_M to match base model quantization
- Modelfiles follow pattern from `models/Modelfile.qwen9b-fast`
- num_gpu=999 forces all layers to GPU
