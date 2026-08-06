# A′ slice-4 Rung-6 PLAN v6 — count-standardization decomposition

**Task:** `1786004998450-f6569bd2`  
**Revision:** `v6_rung6_20260806`  
**Supersedes DEAD:** v5 `61038dc3…` (gate-2 BLOCK `1786010183162`)  
**Review tier:** HIGH  

## Acceptance status (authoritative)

- **STEP-1 code:** dual-accepted (freeze r6 `e061c239…`, co_lead PASS `1786008753155`).
- **STEP-2 code:** **BLOCKED** at gate-2 `1786009786244` (snapshot admission + missing-manifest error path); pending RT2 cures after this plan dual-accept.
- **This MD is NOT activation authority** and must not be cited as dual-accept of STEP-2 or as live-execute clearance.

## v6 delta (plan-gate only)

1. **Operative-currency:** non-historical acceptance strings that still demanded DEAD v4 rewritten to operative v6 / revision-neutral form.
2. **STEP-2 exact battery:** fold seven mandatory cure hostiles into `STEP_2_ASSERTIONS` + `validation_proof.STEP_2.covers` + `test_battery_named`; `test_battery_count_exact` **46→53**.
3. **MD status correction:** STEP-2 not dual-accepted (see above).

## Carried from v5 (timing)

**Runtime-source manifest timing — step-6 semantics govern:**

1. After STEP-2 **code freeze** (gate-1), Claude O_EXCL-mints a **non-operative** four-file runtime-source manifest (scratch, 0444).
2. Freeze handoff to co_lead gate-2 includes manifest abs path + sha256 + per-file map + ORDERED_CONCAT_V0 digest.
3. Manifest becomes **operative only on gate-2 PASS**.
4. Live execute only after dual accept + separate EIGHT-bind `+1 execute`.

**Forbidden restated:** activate/use the runtime-source manifest for live execute before STEP-2 dual accept + gate-2 PASS.  
Non-operative mint-for-review after code freeze is **required**, not forbidden.

All science/branch/claim/runtime-path semantics, SELF_CONTAINED_FOUR_PATH, ORDERED_CONCAT raw bytes, EIGHT binds, claim allowlist, and frozen inputs remain unchanged from the dual-accepted PLAN v4 science body (v5 timing matrix + FCW activate-ban carried). Companion JSON is authoritative for the machine contract.
