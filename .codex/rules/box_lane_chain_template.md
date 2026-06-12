# Box-lane chain packet template (standing)

Task lane: `1781166392598` | Canonical plan r2: `1781272854622`

## Standing invariants (§A–D)

1. **§A Code-currency** — fail-closed rsync+hash preflight (`scripts/box_lane_code_currency_preflight.py`); exit 11 on mismatch; `FETCH_HEAD` binding after fetch.
2. **§B Artifact manifest** — schema `hrm158_box_lane_manifest/v1`; producer+consumer sha256; exit 12 on rsync mismatch.
3. **§C Box receipts** — `execution_host`, `compute_lane`, residency claims; anti-laundering fields required.
4. **§D OVERLAP** — earned only after code-currency pass + artifact sha verify + consumer phase start; watcher `scripts/box_lane_chain_watcher.py` (4B).

## Per-chain packet skeleton (§0–G)

| § | Content |
|---|---------|
| 0 | `chain_id`, HEAD pin, parent sha, single-variable clause |
| A | code-currency preflight — gate 0 |
| B | producer capture (4070 GPU hot-loop) |
| C | artifact rsync + manifest |
| D | consumer chain on box (CPU trace + 1070 cuda_probe) |
| E | model-touching probes with explicit residency |
| F | OVERLAP receipt + pipelining permission |
| G | classifier + terminal stub |

**A1 diff-proof:** rebind-only sections MUST include: `diff vs template §N / source msg id: only tokens X/Y/Z differ`.

**A2 JSON-path receipts:** cite `file:path = value` for numeric/verdict fields (never paraphrase).

## Roots convention (C4)

- `chain_id` basename only
- local+remote: `/home/gabe/claw-code-creditdir/transient_fp_credit/<chain_id>`
- Do not nest absolute `CHAIN_ROOT` under creditdir

## Pinned manifest (C3)

Per-chain `pinned_files_manifest.json` listing every CLI + imported module. Floor set in `box_lane.DEFAULT_FLOOR_PINNED_FILES`; add analyzer surfaces when invoked.

## Validation (V*)

- `--dry-run` must not invoke ssh/rsync (tested)
- Local validation commands timeout-wrapped
