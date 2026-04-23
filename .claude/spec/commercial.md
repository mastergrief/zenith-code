# Commercial Rules

## Product Vision

**Zenith**: a general-purpose intelligence TOOL that runs on consumer
hardware, verifies every answer via compiled programs, gets smarter from
usage without training, and keeps the user in full control. Everything
useful about general AI, nothing dangerous.

## Differentiators (protect these)

| Differentiator | What it is | Why it matters |
|---|---|---|
| **Unified single tensor** | Gemma + HRMs + compiled cards + knowledge DB in ONE .pt | No other system composes trained + compiled + persistent in one forward pass |
| **Level 5 in-Gemma** | Compiled programs inside Gemma's own attention sub-heads | Programs are part of the model, not bolted on — zero overhead |
| **tq4+tq4 hybrid** | TurboQuant for Gemma weights + KV, FP32 for compiled cards | 5B model at 512K context in 8GB VRAM with exact compiled sub-heads |
| **Auto-CALM verification** | 1002 backend functions verify every claim | 100% math benchmark, CPU-checked, not hallucinated |
| **Auto-upgrade loop** | CALM corrections → compile into weights → persist | System gets smarter from usage. Zero training. Usage = moat |
| **Facade / import system** | Module system for compiled neural programs | Domain experts author cards with imports/exports, linker resolves |
| **Compiled reasoning** | Comparison, logic, transitivity as gate-graph programs | Exact logical inference in the model's own attention |
| **Persistent knowledge DB** | Corrections compiled into weights, cross-session | The .pt IS the database. Grows smarter session over session |
| **Fully local** | No cloud, no API, no internet required | Privacy, zero cost, offline-capable |
| **Safety by architecture** | User-controlled, inspectable, reversible, scoped | Alignment without RLHF — the system can't set its own goals |
| **Tier 2/3 augmentation** | Mapping-guided compiled replacements (Tier 2) + from-scratch compiled capabilities (Tier 3) | Tier 1 (preserve) is free — Tiers 2 and 3 are the product. Factorial per-domain scaling; marginal cost of 100th domain ≈ 1st. See `augmentation_thesis.md` |
| **Decode-path facade proliferation** (2026-04-22 receipt) | `calm/llm_computer/recursion.py` + `MetaFacade.from_oracle(fn_name, arity)` + three-gate CALM validation | Adding a compute / text-recall domain is MINUTES, not weeks. 11 facades shipped in one session (6 Level-1 auto + 5 Level-2 meta); 5 hand-written facades in the same session (NumberTheory, NumericEncode, Icd10Recall tier-3, Planner, Icd10 retry). See `compute_facades.md` + `recursion.md`. |
| **Tier-3 decode-path for regulated verticals** | `Icd10RecallFacade` (72,748-code DB, 26/30 on Gemma-fail corpus, `afc0220`) | Hospital / legal / financial / chemical domains: short known-length text recall from a static DB IS tier-3-addressable via decode-path facade. First instance shipped. |

## Safety as a Feature

This is not a concession — it's the pitch:
- Every improvement is a compiled card or key-value fact → **inspectable**
- Previous .pt is the rollback → **reversible**
- Each card fires only on its declared inputs → **scoped**
- System improves only where CALM verifies → **verified**
- No autonomous goal-setting → **user-controlled**
- Runs locally → **private**

Competitors train opaque reward models and hope alignment holds. We
compile verified programs and prove correctness exhaustively.

## Architecture Decisions — Commercial Lens

- **Unified over modular** — one tensor, one forward, one file. Users
  don't manage card files; the substrate IS the product.
- **Local-first** — never add a cloud dependency to the core path.
- **Compiled over trained** — prefer gate-graph compilation (exact, instant,
  free) over LoRA/fine-tuning (approximate, slow, expensive).
- **Self-improving over retrained** — auto-upgrade loop replaces model
  retraining. Usage drives improvement; each session makes the next better.
- **Verified over probabilistic** — when a computation CAN be checked,
  check it. When it can be compiled, compile it.

## Scaling Model

| Hardware | Capacity | Use case |
|---|---|---|
| RTX 4070 (8 GB) | Gemma 4 E4B + 30 domains + knowledge DB | Personal tool |
| RTX 4090 (24 GB) | Gemma 27B + 100 domains | Power user |
| A100 (80 GB) | Gemma 27B + 500 domains + team KB | Team substrate |
| H100 cluster | Gemma 70B+ + thousands of domains | Enterprise |

Team substrate: multiple users share one .pt. Each user's CALM
corrections compile into the same knowledge layer. Collective learning.

## Priority Order for Features

1. **Auto-upgrade pipeline** — wire CALM → compile → persist into the
   harness so it runs automatically. This is the flywheel.
2. **Domain card library** — more compiled ops + HRM specialists per
   domain. Each domain is a feature that sells itself.
3. **Harness integration** — zenith CLI invokes the substrate directly.
   `SubstrateComputer.query()` as the verified-answer backend.
4. **IDE integration** — VS Code extension with substrate as backend.
5. **HRM retraining per domain** — scheduled sampling + domain data.
   15 min per HRM on consumer GPU.

## Business Model Candidates

- **Open core**: substrate framework + base cards free, premium domain
  packages (legal, medical, finance) paid
- **Domain marketplace**: community publishes domain facades with
  compiled ops + HRMs. Revenue share on paid domains.
- **Enterprise on-prem**: team substrate deployed behind firewall.
  Collective learning across the org. Custom domains.
- **Support/SLA**: paid tier for production substrate management.

## Metrics That Matter

- **Autoreg accuracy**: HRM autoregressive accuracy per domain (gate metric, not teacher-forced)
- **Compiled op count**: number of exhaustively-verified compiled programs
- **Domain count**: number of installed domain facades (30 target for v1)
- **Knowledge facts**: accumulated corrections in persistent DB
- **Sub-head utilization**: % of free sub-heads used (capacity planning)
- **End-to-end chain accuracy**: HRM × card = verified answer rate
- **VRAM floor**: 8 GB minimum for useful operation
- **GPU speedup**: 68× at 889M params on RTX 4070 (scales super-linearly)
- **Session-over-session improvement**: does accuracy increase with usage?

## IP and Attribution

- Unified single tensor substrate architecture is original
- Per-sub-head attention partition (Level 5) is original
- Facade/import system for compiled neural programs is original
- Auto-upgrade loop (CALM → compile → persist) is original
- Compiled reasoning primitives inside LLM attention is original
- Depth-compounding via residual channels is original
- Hybrid FP32/tq4 per-layer dispatch is original
- Q6_K PyTorch dequant is a port of llama.cpp (MIT)
- Upstream llama.cpp is MIT; zenith branch patches are original
