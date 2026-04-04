# Training Rules

## VRAM Budget (8 GB RTX 4070)
- Qwen 3 0.6B QLoRA: ~2.5 GB — fits with batch=4, packing=true
- Qwen 3.5 0.8B QLoRA: ~4 GB — requires batch=1, packing=false, seq_len=1024
- Qwen 3.5 4B: too large for QLoRA on 8 GB
- Always stop Ollama before training: `ollama stop <model>` or verify `ollama ps` is empty

## Known Issues
- Qwen 3.5 has 248K vocab which makes fused cross-entropy loss OOM on 8 GB
  - Fix: batch_size=1, max_seq_length=1024, packing=False
- Unsloth compiled cache stored at `unsloth_compiled_cache/` in project root (gitignored)
- Git Bash mangles WSL paths with parentheses in PATH — use `wsl -e bash -c` or write scripts to `/tmp/`

## Dataset Quality
- Claude-authored data is preferred over 9B-generated data (higher quality, faster)
- HuggingFace datasets (TeichAI, Crownelius) provide Claude Opus reasoning traces
- Orchestrator domain is classification (route to specialist), not code generation
- Single epoch prevents catastrophic forgetting (proven by TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning-Distill)

## Export
- GGUF conversion requires llama.cpp (`git clone https://github.com/ggml-org/llama.cpp`)
- Quantize to Q4_K_M to match base model quantization
- Modelfiles follow pattern from `models/Modelfile.qwen9b-fast`
- num_gpu=999 forces all layers to GPU
