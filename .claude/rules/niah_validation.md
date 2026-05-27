---
paths:
  - "scripts/needle_test.py"
  - "agents/config.py"
  - "agents/harness.py"
  - "calm/llm_computer/eval_defaults.py"
  - ".claude/MEMORY/evals/*needle*.md"
  - ".codex/MEMORY/evals/*needle*.md"
---

# Needle-in-Haystack Validation

Effective context for both base models was measured via single-needle,
multi-needle, and distractor NIAH tests at 4K–220K haystack sizes.
Full reports in `.claude/MEMORY/evals/2026-04-07_*_needle_256k_*.md`,
summary in `2026-04-07_summary_needle_comparison.md`.

| Test type | Gemma 4 E4B | Qwen 3.5 4B |
|---|---:|---:|
| Single-needle (21 prompts, 4K–220K) | **21/21** | **21/21** |
| Multi-needle (7 prompts, 5 needles each) | **6/7** (fails 220K 4/5) | **5/7** (fails 180K 4/5, 220K 3/5 + hallucination) |
| Distractor (7 prompts, 4 decoys each) | **7/7** | **5/7** (U-shape dip at 64K/100K — picks wrong decoy) |
| **Total** | **34/35** | **31/35** |

## Key findings

- Both models handle single-needle cleanly through 220K (neither degrades on the easy test).
- Gemma's sliding-window attention protects against the "lost in the middle" failure mode Qwen exhibits at 64K/100K on the distractor test.
- Qwen's worst failure is **hallucination under pressure** (returns `14223-AZURE-MARTEN` when expected is `14223-CRIMSON-EAGLE` — correct number, invented suffix). Gemma's failures are silent omissions, which is a safer mode.
- `agents/compact.py:MODEL_CONTEXT_LIMITS` is the authoritative source of NIAH-validated values (Gemma 200K = 10% safety below the first failure point at 220K; Qwen 130K = safely below the first failure point at 180K).
- **The 256K tests required a local llama.cpp patch** to remove the per-slot training-context cap — see `environment.md` §"Local Tools".

## Related rules

- `environment.md` §"Serving Architecture" + §"Key Constraints" — the VRAM/context configs these numbers were measured against
- `architecture.md` §"Agent System" — `MODEL_CONTEXT_LIMITS` + 89% safe-ctx compaction margin
- `CLAUDE.md` — top-level index
