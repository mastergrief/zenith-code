---
paths:
  - "calm/llm_computer/**"
  - "scripts/probe_gemma*.py"
  - "scripts/dt_install_probe.py"
  - "scripts/*debug_probe.py"
  - "scripts/test_*_probe.py"
  - "scripts/r*_probe.py"
  - "scripts/*sweep*.py"
  - "scripts/*ablation*.py"
  - "scripts/*planner*.py"
  - "scripts/*hospital*.py"
  - "scripts/*icd10*.py"
  - "RESEARCH/**"
---

# Augmentation Thesis — explicit structure over opaque scale

Settled positions for the current arc. Future sessions treat these as
baseline context, not re-argue them.

> Historical receipts + Gemma-substrate first instantiation detail:
> `MEMORY/atlas/augmentation_thesis_arc.md`.

## Core thesis

Frontier-like reliability on structured work comes from **explicit,
auditable structure** composed on a small persistent model — not from
opaque scale.

Three **current carriers** on the active arc:
1. **CALM** — live verified computation (CPU oracle on compiled or
curriculum outputs)
2. **hrm-158 curriculum** — progressive checkpoint banking with
90/90 acquire/retain gates on auditable finite supports
3. **Ternary training stack** — ternary effective weights with
integer-dominated persistent train-state (toward sub-2-bit persistent;
the banked win is FP-master-free for the eligible bulk — see
`ternary_hybrid_stack.md` 3-ledger)

**Gemma + substrate cards** were the **first shipped instantiation**
of the same pattern (routing → compiled correction → verification
hook). That lane is **legacy/adjacent unless reopened**; the active
default is native HRM-Text-1.58 (`hrm-158.md`). Arc detail: atlas
§"Gemma-substrate instantiation".

## Three-tier framework

| Tier | General meaning | Maps onto curriculum (hrm-158) | Moat |
|---|---|---|---|
| **1 — Preserve** | Don't touch what already works | Banked parent capability: replay+pc on acquired priors; no intervention on surfaces the parent already holds | none |
| **2 — Augment weak** | Surgical correction at a failing site | Targeted finite-support slice on a classified failing surface/template; additive repair without whole-model retrain | high |
| **3 — Plug missing** | New capability where no prior exists | New auditable support when parent has zero relevant prior; procedure-supervised curriculum when answer-only memorizes | highest |

**Convention note:** substrate install tiers (CardSlot, VerificationHook,
hub injection) **map onto** these curriculum semantics by analogy — see
atlas for Gemma-specific mechanics. Do not treat the mapping as a measured
identity or banked doctrine.

## Classify before build

Run per-head ablation AFTER the layer sweep to classify. Compile
decision flows from the classification.

| Shape | Action |
|---|---|
| **Concentrated** | 1-2 surgical replacements at the dominant site |
| **Cooperative** | 3-4 replacements, additive composition |
| **Diffuse** | NOT compilable at attention level; side-channel or FFN probing |
| **Deep-diffuse** | Not distillable at known loss spaces; pivot to additive output correction |

**Rule**: never attempt attention-level compilation without
classifying via per-head ablation first. See atlas for Gemma-specific
markers and failure-mode lineage.

## Compositional hypothesis

Complex capabilities are short walks through a sparse set of
primitive mechanisms. **Individual units are the atoms, not layers.**
Shared hub mechanisms carry task-neutral reads; task-specific routing
activates different subsets.

One compiled replacement at a shared hub can serve N capabilities
simultaneously — when the same mechanism serves multiple tasks via
task-specific routing, replacing it once is N-for-1 compilation ROI.

## Beyond "structured" tasks

The pattern extends to any task with operationalizable quality criteria.

**Verification keeps long context usable** — models degrade past long
horizons because errors compound silently; per-step verification
rejects bad steps before they propagate.

### Unstructured ≠ incompilable

The "unstructured" label is really "structure humans don't explicitly
articulate." Operational criteria emerge as soon as you compare a
good answer to a bad one. Each becomes a verifier. Good
interpretive/creative work decomposes to: **retrieval** (DB) +
**transformation** (compiled correction) + **verification** (oracle
or exact checker).

### Auditable / reversible / private = the *better* product

For compliance-adjacent industries the explicit-structure stack is
the better product, not merely the cheaper one: **auditable** (facts
and compiled rules, not opaque weights), **reversible** (remove any
add-on cleanly), **private** (local, no API), **correct** (verified
by construction on compiled tasks). The business question isn't "as
smart as GPT-4 in general?" — it's "which product do regulators and
auditors trust?"

## Anti-skepticism — what NOT to relitigate

| Objection | Settled counter |
|---|---|
| "Works only for structured/numeric tasks" | Poetry + analogies + long-horizon planning counter-examples; constraint gates generalize. |
| "Factual recall needs frontier models" | Diffuse circuits — compilable via side-channel DB with verified retrieval. |
| "Retraining needed for each domain" | Tier-2 stacking takes hours/days per domain on the substrate lane; curriculum slices take bounded windows. No whole-model retraining. |
| "Frontier capabilities can't be matched" | Frontier advantages = scale-of-retrieval + verification + structure. All three compile or bank. |
| "Doesn't scale — you'll cap on engineering" | Factorial scaling argument. Per-domain cost is flat once the stack exists. |
| "Verification is the bottleneck" | CALM verifies on benchmark; compiled verifiers are exact by construction. |

When probing a new capability, classify first. Do not re-derive the
thesis mid-task.

## Related rules

- `hrm-158.md` — active curriculum lane (native HRM-Text-1.58)
- `ternary_hybrid_stack.md` — ternary economics and 3-ledger accounting
- `calm.md` — verification layer (CPU oracle)
- `Substrate.md` — install mechanics (legacy/adjacent unless reopened)
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `capability_gain.md` — measurement discipline (raw + user-facing)
- `workflow.md` — iteration discipline (hypothesis → test → commit)
- `MEMORY/atlas/augmentation_thesis_arc.md` — empirical basis + Gemma instantiation
