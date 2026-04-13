# Commercial Rules

## Product Vision

**Zenith**: a local AI coding assistant that runs on consumer hardware,
verifies every answer on CPU, and gets smarter from usage without training.

## Differentiators (protect these)

| Differentiator | What it is | Why it matters |
|---|---|---|
| **tq4+tq4 hybrid** | Custom TurboQuant quantization for weights + KV cache | 4B model at 512K context in 8GB VRAM — no competitor does this |
| **Auto-CALM** | Transparent compute verification | Every claim CPU-checked, not hallucinated. 100% math benchmark |
| **Self-learning** | Learned patterns compound from usage | System improves without training runs. Usage = moat |
| **Modular backends** | Drop-in Python compute modules | Domain experts add backends, model gets instantly smarter |
| **Fully local** | No cloud, no API, no internet required | Privacy, zero cost, offline-capable |

When building features, ask: **does this strengthen a differentiator?**
If not, deprioritize it.

## Architecture Decisions — Commercial Lens

- **Modular over monolithic** — every component should be independently
  useful, replaceable, and extensible. Users and enterprises add backends.
- **Local-first** — never add a cloud dependency to the core path.
  Cloud is an optional enhancement, not a requirement.
- **Self-improving over trained** — prefer learned patterns (CPU, instant)
  over LoRA (GPU, expensive). Training is supplementary, not primary.
- **Verified over probabilistic** — when a computation CAN be checked,
  check it. Verified answers are the brand promise.
- **Standard interfaces** — OpenAI-compatible API, GGUF models, Python
  backends. Don't invent proprietary formats that lock users in.

## Business Model Candidates

- **Open core**: harness + base backends free (MIT/Apache), premium
  backends and enterprise features paid
- **Backend marketplace**: community publishes domain backends,
  revenue share on paid ones
- **Enterprise on-prem**: custom backends for company domains
  (finance, medical, legal), deployed behind their firewall
- **Support/SLA**: paid support tier for enterprises running in production

## Priority Order for Features

1. **User-facing quality** — the harness must feel polished, not research-y.
   Error messages, help text, onboarding, responsiveness.
2. **Backend coverage** — more domains = more "smart" without training.
   Each backend is a feature that sells itself.
3. **Self-learning loop** — the longer someone uses Zenith, the better it
   gets for their workflow. This is retention and moat.
4. **IDE integration** — VS Code extension is the path to adoption.
   Terminal REPL is for power users only.
5. **Model quality** — better base models (larger, fine-tuned) improve
   everything. But Auto-CALM means even weak models produce correct results.

## What NOT to Build (Yet)

- Cloud hosting / SaaS — stay local-first, don't split focus
- Mobile / embedded — desktop + server is the market
- Training infrastructure — the point is you DON'T need it
- Multi-language harness rewrites — Python is fine, optimize later
- Custom model architectures — ride llama.cpp and upstream improvements

## IP and Attribution

- Upstream llama.cpp is MIT. Our zenith branch patches (tq4 fusion,
  Gemma GLU fix, OP_TIMING) are original work.
- Auto-CALM, modular backends, self-learning loop are original.
- CALM engine architecture (LLM sequences, CPU computes, TMR verifies)
  is original.
- Training data pipeline uses HuggingFace datasets under their licenses.
  Hand-written examples are original.
- Keep Co-Authored-By lines in commits for transparency.

## Metrics That Matter

- **Benchmark score**: Auto-CALM + precompute on 40-problem suite (currently 40/40)
- **Backend count**: number of verified compute + knowledge functions (currently 500 across 64 backends)
- **Learned patterns**: size and hit rate of learned_patterns.jsonl
- **VRAM floor**: minimum GPU memory for useful operation (currently 8GB)
- **tok/s**: inference speed on target hardware (currently ~47 tok/s on RTX 4070)
- **Time-to-value**: how fast a new user goes from install to useful output
