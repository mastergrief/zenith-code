---
paths:
  - "agents/distill/**"
  - "agents/model_swap.py"
  - "agents/specialist_coordinator.py"
  - "scripts/self_distill_synth.py"
  - "scripts/eval_base_models.py"
---

# Distillation Pipeline Rules

Training-specific operational reference for `agents/distill/`.
Not a substrate / tracing rule — this covers traditional
fine-tuning / distillation of base + specialist models.
Consult when actively training, not on general sessions.

## Current Status

- **0.8B reasoning base**: trained (3 epochs, loss 1.106), eval:
  format learned but substance wrong — model too small.
- **4B reasoning base (Qwen)**: trained on Colab A100 (910
  examples after re-filter pass, 3 epochs),
  exported to GGUF Q5_K_M, serving via llama.cpp. Earlier eval:
  3/5 PASS with thinking enabled (race condition, OOMKilled,
  architecture pass; React and security partial). **Subsequent
  5-prompt A/B vs stock Gemma 4 E4B scored
  fine-tuned Qwen 0/5** — the React and security failures were
  correctness bugs (hallucinated Node.js `beforeOOM` API, broken
  Postgres `FOR UPDATE SKIP LOCKED` queue, regex-on-hostname SSRF
  check). See `.claude/MEMORY/evals/2026-04-07_qwen4b_vs_gemma4_e4b.md`.
- **Gemma 4 E4B (stock)**: validated as alternative base.
  Beats fine-tuned Qwen 5/0 on the same coding eval
  without any fine-tuning. NIAH effective context: 200K (vs
  Qwen's 130K). Multimodal (vision projector available). GGUF on
  disk at `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB). Not
  yet fine-tuned on the distillation dataset.
- **Hot-swap infrastructure**: IMPLEMENTED (`agents/model_swap.py`
  + `SpecialistCoordinator` hot-swap mode). `LlamaServerManager`
  handles kill+restart swap cycles (~5–15s depending on model
  size). Swap cost on a warm page cache is mostly PCIe transfer
  time.
- **Specialists**: not yet trained — both base models are ready,
  specialist GGUFs don't exist on disk yet. When they do,
  `SpecialistCoordinator` auto-detects and switches to hot-swap
  mode.

## Pipeline Scripts

- `config.py` — Domains, model names, QLoRA params, paths
- `generate.py` — Teacher (9B) generates JSONL training data.
  CLI: `python -m agents.distill.generate --domain python`
- `train_base.py` — Stage 1: 0.8B reasoning base (local).
  Qwen 3.5 0.8B + QLoRA + `train_on_responses_only`, 3 epochs
- `train_4b_cloud.py` — Stage 1: 4B reasoning base (cloud).
  Qwen 3.5 4B + QLoRA, requires 40GB+ VRAM (A100)
- `train_4b_colab.ipynb` — Colab notebook for 4B training
- `train.py` — Stage 2: domain specialist training. Auto-uses
  reasoning base if available
- `export.py` — Convert merged model → GGUF → Ollama Modelfile →
  `ollama create`
- `validate.py` — A/B compare specialist vs base using 9B as judge
- `fetch_datasets.py` — Download Claude reasoning datasets from
  HuggingFace (nohurry, TeichAI, Crownelius)
- `filter_reasoning.py` — Tiered keyword filtering + dedup + merge
  hand-written with HuggingFace data

## Specialist Domains

| Domain | Ollama Name | Focus |
|--------|-------------|-------|
| orchestrator | specialist-orchestrator | Task routing/classification |
| typescript | specialist-ts | React, Node, TS, Next.js |
| python | specialist-py | FastAPI, Django, pytest |
| rust | specialist-rust | Ownership, tokio, serde |
| devops | specialist-devops | Docker, K8s, Terraform |
| reviewer | specialist-reviewer | Security, bugs, perf |

## Training Data (`agents/distill/data/`, gitignored except hand-written files)

- `claude_reasoning.jsonl` — 910 merged examples (re-filtered pass
  over HuggingFace + hand-written sources; earlier 1,339 count
  reflected a pre-filter merge)
- `coding_reasoning_claude.jsonl` — 547 hand-written coding
  reasoning examples (committed). Includes React + security examples
  targeting Qwen eval gaps.
- `claude_reasoning_filtered.jsonl` — 832 filtered HuggingFace
  examples (intermediate)
- `claude_reasoning_prefilter.jsonl` — backup of pre-filter merged
  data
- `orchestrator.jsonl` — 252 routing examples (130 original + 121
  Claude-authored)
- `orchestrator_claude.jsonl` — 121 Claude-authored routing
  examples (committed)
- `python.jsonl` — 25 examples (9B-generated)
- `typescript.jsonl` — 39 examples (9B-generated)
- `rust.jsonl` — 53 examples (9B-generated)

## Training Commands

```bash
# Stage 1: 0.8B reasoning base (local, 8GB VRAM)
PYTHONPATH=. python3 -m agents.distill.train_base

# Stage 1: 4B reasoning base (cloud, 40GB+ VRAM)
# Use train_4b_colab.ipynb on Google Colab with A100
# Or: python3 train_4b_cloud.py on RunPod/Lambda

# Stage 2: Specialist (on top of reasoning base)
PYTHONPATH=. python3 -m agents.distill.train --domain orchestrator

# Filter + merge reasoning data
PYTHONPATH=. python3 -m agents.distill.filter_reasoning        # filter only
PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge # filter + merge hand-written
```

## Training Philosophy (distillation-specific)

**Data quality > data quantity > model size > training tricks.**

1. Write high-quality examples — one good Claude-authored example
   teaches more than ten 9B-generated ones
2. Train on responses only — don't waste gradients learning to
   predict prompts
3. Match domain to task — coding reasoning data for coding models,
   routing data for routing models
4. Filter aggressively — removing bad data improves results more
   than adding mediocre data
5. 3 epochs on curated data — diverse enough (1,320 unique topics)
   to avoid memorization; 1 epoch underfits

## Related rules

- `training.md` — broader training rules (HRM, PT, substrate-native
  cards, quantization). Distillation-specific details live here;
  shared training wisdom in `training.md`.
- `calm.md` — Auto-CALM auto-training data collection (generates
  distillation-compatible JSONL from corrections)
