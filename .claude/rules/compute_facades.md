---
paths:
  - "calm/llm_computer/facades/**"
  - "calm/llm_computer/recursion.py"
  - "calm/llm_computer/program_builder.py"
  - "scripts/*facade*.py"
  - "scripts/*planner*.py"
  - "scripts/*hospital*.py"
  - "scripts/*icd10*.py"
  - "scripts/*metafacade*.py"
---

# Compute Facades — legacy/adjacent unless reopened

> Historical receipts + operational detail: `MEMORY/atlas/compute_facades_arc.md`.

**Lane status:** legacy/adjacent unless reopened. Active default = native HRM-Text-1.58 (`hrm-158.md`).

**Decode-path vs CardSlot:** ship decode-path first for oracle-backed
domains; CardSlot only for genuine trained-recall — see atlas §"R22b
calibration reference".

## Related rules (stubs)

- `recursion.md` — Level-1 `FacadeSpec` + Level-2 `MetaFacade` (legacy/adjacent unless reopened)
- `capability_gain.md` — two-measurement A/B discipline (legacy/adjacent unless reopened)
- `embed_intelligence.md` — step-through bias mechanics (legacy/adjacent unless reopened)
- `MEMORY/atlas/compute_facades_arc.md` — archaeology + receipts
