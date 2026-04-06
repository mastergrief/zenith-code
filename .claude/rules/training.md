# Training Rules

## VRAM Budget

### Local (8 GB RTX 4070 Laptop)
- Qwen 3 0.6B QLoRA: ~2.5 GB — fits with batch=4, packing=true
- Qwen 3.5 0.8B QLoRA: ~4 GB — requires batch=1, packing=false, seq_len=1024
- Qwen 3.5 4B QLoRA: **OOM on 8 GB, 16 GB, and 15 GB** — 248K vocab CE loss is too large
- Always stop Ollama before training: `ollama stop <model>` or verify `ollama ps` is empty

### Cloud (Colab Pro A100 40GB)
- Qwen 3.5 4B QLoRA: fits with batch=1, seq_len=1024, packing=false
- Use `agents/distill/train_4b_colab.ipynb` or `train_4b_cloud.py`
- Cost: ~$0.50-1.00 per training run (~30-40 min on A100)

## Priority Order
- **Data quality > data quantity > model size > training tricks.** One hour writing 20 high-quality examples beats hours of hyperparameter tuning.
- **Model size matters for correctness.** 0.8B learned the `<think>` format but gave wrong technical answers. 4B should fix this.
- Each example should demonstrate the *reasoning process* (`<think>` block), not just the answer
- Match the training domain to the task: coding data for coding models, routing data for routing models

## Training Best Practices
- Always use `train_on_responses_only` — masks instruction/prompt tokens so loss is only computed on the model's generated responses
- ~1,300 curated examples with 3 epochs works well (confirmed: 0.8B run 2, loss 1.106)
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` is the best-filtered Claude reasoning dataset on HuggingFace
- 3 epochs on curated, diverse data — 1 epoch underfits, more epochs on small datasets = memorization
- Filter training data before use: `python -m agents.distill.filter_reasoning --merge`
- Filter aggressively — removing bad data improves results more than adding mediocre data

## Dataset Quality
- Claude-authored/hand-written data >> 9B-generated data (higher quality, more consistent)
- HuggingFace datasets (nohurry, TeichAI, Crownelius) provide Claude Opus reasoning traces — filtered to 832 examples
- Hand-written data committed to repo: `coding_reasoning_claude.jsonl` (488 examples), `orchestrator_claude.jsonl` (121 examples)
- Merged training file: `claude_reasoning.jsonl` (1,320 examples = 832 HF + 488 hand-written)
- Filter pipeline: tiered keyword matching (1 strong keyword + 2 general, OR 5+ general, OR code blocks), dedup by first 60 chars, think-block minimum lengths
- Filter out: hallucinated facts, non-technical content, NLP benchmark patterns, junk (<200 char responses)
- Specialist datasets are still small (25-53 examples) — need expansion after 4B base is ready

## Known Issues
- **Qwen 3.5 248K vocab**: fused cross-entropy loss OOMs on anything under 40GB VRAM for 4B, under 8GB for 0.8B
  - 0.8B fix: batch_size=1, max_seq_length=1024, packing=False
  - 4B fix: use cloud GPU (Colab A100)
- Unsloth compiled cache stored at `unsloth_compiled_cache/` in project root (gitignored)
- Git Bash mangles WSL paths with parentheses in PATH — use `wsl -e bash -c` or write scripts to `/tmp/`
- No Thunderbolt/eGPU on Acer Nitro AN17-42 — cloud GPUs required for 4B+ training
- RunPod SSH proxy unreliable from WSL2 — use web terminal or Colab instead

## Export & Serving
- llama.cpp built at `~/llama.cpp/build/bin/` (CPU-only build, handles GGUF conversion)
- `llama-quantize`: convert FP16 safetensors → GGUF quantized (Q5_K_M recommended for 4B)
- `llama-server`: serve with OpenAI-compatible API, supports KV cache quantization
- For Ollama: Modelfiles in `models/`, use `ollama create` with Windows paths from WSL (`cmd.exe /c "ollama create ..."`)
- Quantization target: Q5_K_M for 4B (best quality/size ratio at 64K context with Q4 KV cache)
- num_gpu=999 forces all layers to GPU
