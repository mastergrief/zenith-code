# A′ slice 4 Rung-3 PLAN v6 — residual classification (receipts-only)

**Task** `1785950193161-4f423eb4` (successor; parent `1785931309579-4f068343`)  
**Superseded child** `1785949293431-20897287` — SUPERSEDED / lineage only  
**Dispatch** `1785950352907` · **Tier** HIGH  
**Supersedes** v0–v5 **DEAD**  
**JSON** sha `bddf41c768b48efc4346618c3c9f9f8f5285eacfcf1159de728c5b4743fb96ca`

> Product: **RESIDUAL_CLASSIFICATION_RECEIPTS_ONLY** (factorized; zero-GPU)

## Why v6
gate-2 REVISE on v5 (`1785950314545`): everything cleared except plan still bound to cancelled task id. Rebind all current authority to successor `1785950193161-4f423eb4`.

## Cure
- `task_id` / `board.child_task_id_current` / receipt-sink / ownership → `1785950193161-4f423eb4` only.
- Cancelled `1785949293431-20897287` appears only in supersedes/lineage labelled SUPERSEDED — never current/owner/sink.
- board.decision_contract_currency: full 64-hex v6 sha + CONTENT_DIGEST + freeze-manifest sha posted by claude as contract-currency addendum task_update on `1785950193161-4f423eb4` immediately after gate-1 freeze (board append-only; decision_contract create-time-only).

## Science content
byte-invariant vs v5: prereg / frozen_inputs / claim_boundary / claim_effect / product_name / validation_argv_frozen / implementation_contract / workflow sequence incl. step-7 activation gate.

## After dual-accept
STEP-1 +1 → dual accept → STEP-2 +1 → dual accept → **+1 execute live terminal** → Claude live terminal. No mechanism mint.
