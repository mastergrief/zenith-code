---
paths:
  - "bin/**"
  - "agents/model_swap.py"
  - "agents/harness.py"
  - "scripts/llama_cpp_patches/**"
  - "scripts/bench*.py"
  - "scripts/gpu_*.py"
  - "scripts/validate_gemma4_vs_llamacpp.py"
---

# Environment — Hardware, Serving, Tooling, Accounts, Constraints

All "what's installed / what's available" facts in one place. Update
here when hardware, GGUF paths, accounts, or VRAM budgets change.

## Hardware

**Laptop (Acer Nitro AN17-42) — primary trainer + serving:**

- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM, Ada, Tensor Cores) — no Thunderbolt/eGPU support
- iGPU: AMD Radeon 780M (display only, no CUDA)
- RAM: 32 GB DDR5 5600MHz
- WSL2: Ubuntu 24.04 with GPU access
- Training (local): Unsloth 2026.4.2 + PyTorch 2.10.0+cu128

**The box — audit/probe helper lane (separate Linux machine, over LAN):**

- GPU: NVIDIA GTX 1070 (8 GB VRAM, Pascal CC 6.1) — **no Tensor Cores**; consumer-Pascal native FP16 is ~1:64 of FP32, so its lane is integer/ternary inference + small-model FP32, **NOT** FP16/bf16 training.
- Reached over LAN via a multiplexed connection; the "audit box" is a plain rsync dir (not git) — see `hrm-158.md` §Validation pre-launch code-currency check.
  Science-chain roots: `/home/gabe/claw-code-creditdir/transient_fp_credit/<chain_id>` (template: `claw-code-hrm-text-158/.codex/rules/box_lane_chain_template.md`).
- Role: runs the producer/consumer audit-watcher probe bundle so the 4070 trains uncontended. The native ternary path runs after the portability fix; still measure sm_61 train throughput before committing *training* (vs inference) to it.

**Cloud:**

- Training: Google Colab Pro A100 (40GB)

## Serving Architecture

**llama.cpp (primary)** — Gemma 4 E4B via full GPU:

- **Production GGUF**: `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, TurboQuant tq4, 132-byte block alignment for 4-byte aligned CUDA loads). **This is what CALM runs on.**
- **Alternative GGUFs**: `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock Q5), `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, fine-tuned Q5)
- **TurboQuant tq4 KV cache**: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`. 4.125 bpw, 16-level Lloyd-Max codebook, Pi rotation (seed=42, 256×256 orthogonal). 132-byte blocks (128 qs + 2 d + 2 pad for 4-byte aligned uint32 loads). **Old 130-byte GGUFs are incompatible — re-quantize.**
- Context: **512K** with tq4 KV (~5.0 GB weights + ~2.0 GB KV = ~7 GB VRAM). 48K thinking budget (`EFFORT["max"]["max_tokens"]=49152`). Auto-CALM + harness share the same server.
- `--parallel 1` required — without it, llama-server splits `ctx_size` across 4 default slots
- Launch: `llama-server -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf --ctx-size 524288 --parallel 1 --cache-type-k tq4_k256 --cache-type-v tq4_k256 -ngl 999 --port 8080`
- **Decode perf (median-of-5):** 25.02 tok/s tq4+graphs / 33.35 tok/s fp16+graphs / 7.14 tok/s tq4 no-graphs. ~60% / 79% / 17% of llama.cpp ~42 tok/s. Historical "90% of llama.cpp" claim reserved for hardware/driver state that may not match current bench — rebench to confirm if comparing to that baseline.
- Hot-swap: `agents/model_swap.py:LlamaServerManager`. `/swap gemma` / `/swap qwen` in harness.

**Ollama (fallback)** — stock models, quick testing:

- Pulled models (verified via `curl -s localhost:11434/api/tags`): `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b`, `qwen3:4b`, `qwen3:8b`, plus custom Modelfiles `qwen4b-fast:latest`, `qwen9b-fast:latest`, `reasoning-base:latest`
- Custom Modelfiles in `models/`: `Modelfile.qwen9b-fast` (2048 ctx), `Modelfile.qwen4b-fast` (8192 ctx), `Modelfile.reasoning-base` (32K ctx)
- Kill Windows Ollama to free VRAM: `taskkill /IM ollama.exe /F`

tq4 kernel internals + TurboQuant block formats + Triton kernel
receipts live in `turboquant.md`.

## Local Tools

- **llama.cpp**: built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070). **Branch `zenith`** carries 3 custom commits beyond upstream (see `git log` for SHAs):
  - **OP_TIMING**: per-op/per-shape cudaEvent timing diagnostic. Enable: `cmake -DGGML_CUDA_OP_TIMING=ON`, `GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OP_TIMING=1`
  - **Gemma gate+up ordering fix**: upstream fusion check at `should_fuse_mul_mat` rejected Gemma's reversed ordering — GLU fusion was never firing on any Gemma quant type upstream. Worth upstreaming.
  - **Fused gate+up+GLU tq4 kernel**: `k_mmvq_tq4_k256_fused_preload_glu` in `mmvq-tq4.cu`. +0.68% avg (structural win — ships one Pi@x precompute, eliminates GLU kernel launch).
  - `llama-quantize`, `llama-server` — standard tools
  - **Local patch** at `tools/server/server-context.cpp:763-766` — comments out `n_ctx_slot = n_ctx_train` for >128K context on Gemma. Re-apply after `git pull`.
  - **5 mmvq-tq4 rounds reverted** (SHFL LUT, NB template, 4-row/block, PiX memoization, 2-way accumulator). Kernel is at a deep local optimum. See `SESSION_HANDOFF.md` ruled-out log.
- **Unsloth**: 2026.4.2 + PyTorch 2.10.0+cu128 (for local 0.8B training)
- **Serena MCP**: installed at `/home/gabe/serena-fork`, configured for this project

## Cloud Accounts

- **RunPod**: API key in `.env.local`, `runpodctl` installed, MCP server configured in `~/.claude.json`
- **Google Colab Pro**: 100 compute units, Colab MCP configured in `~/.claude.json`

## Key Constraints

- **8 GB VRAM**: production default is Gemma 4 E4B tq4 + tq4 KV at **512K context** (~7 GB total, see Serving Architecture above). Historical Q4-KV + Q5_K_M configs fit at **256K context** — Qwen 3.5 4B Q5 uses ~7.3 GB, Gemma 4 E4B Q5 uses ~6.7 GB (sliding-window attention makes Gemma's KV cache dramatically smaller at long context). 9B Q4 fits at 2K (Ollama). 0.8B FP16 fits at 32K.
- **Both 4B bases are trained at 256K context** (Qwen 3.5 4B and Gemma 4 E4B, verified via GGUF metadata `n_ctx_train=262144`). Earlier notes had Qwen at 32K — that was wrong.
- **Qwen 3.5 4B QLoRA**: 248K vocab CE loss OOMs on anything under 40GB VRAM. Must use cloud GPU (Colab A100).
- **Qwen 3.5 0.8B QLoRA**: fits locally at batch=1, seq_len=1024, packing=false
- **WSL2 + Windows Ollama**: both can run Ollama. Don't run both simultaneously — VRAM conflict. Prefer WSL-native.
- **No eGPU**: laptop has USB-C 3.2 but no Thunderbolt/USB4. Cloud GPUs for 4B+ training.

## Related rules

- `turboquant.md` — tq4 block format, Triton kernels, fused flash-attn decode, per-kernel bench receipts
- `niah_validation.md` — `MODEL_CONTEXT_LIMITS` source of truth (200K Gemma, 130K Qwen)
- `harness.md` — `zenith` launcher flags + env vars that wrap llama-server
- `training.md` — Substrate FP32 hosting layer VRAM budget + Substrate eval defaults
- `distillation.md` — cloud vs local training decision table
- `CLAUDE.md` — top-level index
