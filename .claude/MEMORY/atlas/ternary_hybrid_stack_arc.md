# Ternary-Hybrid Training Stack — receipts arc

Query-triggered receipts for `.claude/rules/ternary_hybrid_stack.md` (the
FP-free / sub-2-bit-persistent ternary-training research lane). The rule file
holds current invariants only; per-run measurements, attribution receipts,
mechanism nulls, chain-head shas, and msg IDs land here + on the ai-room board
+ commit log.

## Per-run / per-mechanism receipts

(Append dated entries as runs land. Each entry: run/slice, mechanism under
test, the 3-ledger numbers, verdict taxonomy class + sub-reason, banked-sha
before/after, key artifact shas, ai-room msg IDs, and what the null ruled out.
Cite commits here; reference this atlas from the rule with a one-line pointer.)

### 2026-06-09 — HRM-158 sub-2-bit within-tie-band credit-ranking arc (branches 1-5)

Question: can the ternary learner rank optimizer credit within the
`voteabs=4|marginabs=4` tie band, toward a persistent sub-2-bit eligibility
representation?

Path note: this receipt landed in the main-repo canonical atlas because the HRM
fork lacks `ternary_hybrid_stack.md`; HRM code/receipts remain on
`origin/feature/hrm-text-1.58` at commits `ddb0b0a`, `08bbb3fd`, and
`6925f0b1`.

- B1 signed-residual: degenerate (offline).
- B2 topology: `structural_only_no_signal` (offline).
- B3 first-order magnitude q5 (`ddb0b0a`):
  `activation_credit_ambiguous_no_branch`. It appeared to show
  `first_order_magnitude_insufficient`; `F_magq5_only` had regret_spread
  `.651/.741`, regret_capture `.383/.458`, bucket_fraction `.172/.192`, and
  null guards `1.0`.
- B4 second-order Taylor / diagonal-Fisher q5 (`08bbb3fd`): ambiguous.
  `F_taylor_benefit_q5` was bit-identical to B3; `candidate_delta_weight`
  had constant magnitude `{+/-0.0331}` (single-step ternary flips make
  `delta_w^2` constant, so second-order curvature was void). SNR and
  diag_fisher ablations were non-rescuing and failed the regret-capture null
  guard.
- B5 tracked ceiling audit, no GPU (`6925f0b1`): INVERSION. Raw continuous
  first-order benefit `-g * delta_w` (approximately `abs_grad_proxy`,
  approximately Taylor here) ranks oracle-best near-perfectly: AUC
  `.9951/.9877`, oracle-best top1/top3/top5 = `1/1/1` on both seeds
  (direction fixed on seed43, out-of-sample on seed29). Two-loss decomposition:
  continuous-to-q5-ordinal loss `.084/.079` is small (`q5_bin_index` AUC
  `.911/.909`, so ordinal quantization preserves rank), while
  q5-ordinal-to-receipt-family loss `.330/.292` is large. Classification:
  `receipt_family_bucket_tiebreak_loss` /
  `ordinal_signal_survives_q5_but_current_family_decision_collapses_it`.
  Leak upper bound (`neg local_loss_delta`) reached AUC `1.0` and is tagged
  non-decision.

Root cause: B3/B4 "credit insufficient" were sub-2 measurement artifacts. The
eligibility signal exists, is cheap (first-order grad, already captured), ranks
the oracle-best move, and survives low-bit ordinal quantization (`.91`). The
large collapse (`.58`) is the current receipt-family bucket + current-rank
decision/tiebreak rule, not bit-width.

Claim boundary: this is strong positive mechanism evidence that the signal
exists. It is NOT a learner/runtime success claim, NOT an eligibility build,
and NOT a full-sub2/readiness claim.

Redirect: the next branch is sub-2 decision/tiebreak encoding that preserves
the raw first-order ordinal order. Do NOT pivot to joint/interaction search or
more second-order scalar algebra on this evidence.

### 2026-06-14 — Receipt-family frontier closed; pivot to int16 vote-acc (B5b / H1 / B0)

Receipt-family AUC track parked after CPU counterfactual nulls on Branch4
seed43+seed29 historical receipts. Persistent-width lane (int16 vote-acc
reduction) remains open.

- **B5b** (`BRANCH_TIEBREAK_STILL_COLLAPSES`, commit `96ac7405` on
  `origin/feature/hrm-text-1.58`): within fixed `F_taylor_benefit_q5` bucket,
  no intra-bucket tiebreak key recovers receipt-family AUC >0.75
  (current_rank / ternary_L3 / q5_L5 / raw_fp ~0.60–0.65 on both seeds).
  Tiebreak key is NOT the collapse lever.
- **H1** (`NO_BUCKET_ARM_RECOVERS`, commit `b620c3c4`, 9 pre-registered arms):
  swapping bucket feature/encoder does not recover — best
  `taylor_L3`/`eligibility_L3` 0.680/0.657 <0.75 both seeds. Bucket
  representation is NOT the collapse lever.
- **B0** (`MEASUREMENT_STATE_EXISTS_AND_HEADROOM`, commit `d140f46`): int16
  vote-acc headroom to `w_min=6` on recorded-row single-trace fixture
  (16→6 bits; NOT sub-2 — w4/w3/w2 fail crossing invariance). Threshold
  mismatch `derived=1` vs attested `10` surfaced (replay on attested 10).

Claim boundary: B5b/H1 are informative nulls on decision/receipt-family
discrimination — they do NOT close the persistent-width lane. B0 sizes the
prize for int16 vote-acc reduction; does not bank a sub-2 or GPU verdict.

Receipts: plan-dev implementation receipts `1781446970587` (B5b),
`1781448090254` (H1), `1781449527672` (B0); dual impl review PASS claude +
co_lead on each.

Receipts:

- Manual row audit / board synthesis: Claude `1780989612006`,
  `1780990002366`, `1780990199645`; co_lead independent reproduction
  `1780990139339`.
- Tracked audit receipt: `1780991880860`; commit `6925f0b1`; CLI output sha
  `5f232a362be43d3849043c7384f773b448315e795417977d11fb5c3a40ef3317`;
  tracked audit module
  `calm/hrm_text_158/native_full_stack/activation_credit_ceiling_audit.py` and
  CLI `scripts/hrm_text_158_activation_credit_ceiling_audit.py`.

### 2026-06-24 — R5.1 W5 branch-C decision-parity Tier-1 smoke → informative NULL

Question: does dense W5 byte-packed vote-accumulator ([-15,15], 5-bit signed)
preserve decision-parity dynamics vs W6 oracle on banked R4.1 parent geometry
during a 10-step all-bitlinear diagnostic?

Path note: HRM code/receipts on `origin/feature/hrm-text-1.58` at commit
`785402b` (W5 codec + decision-parity classifier); packet
`artifacts/consensus_prep/r5_1_w5_decision_parity_tensorwide_gpu_launch_packet_v1.json`
sha `1f79d2a8cd3b4e0d8e6d04a3f034d55e16bfbd8c502f44e0ab7a67e58358763b`.

- **Verdict:** informative NULL — dense W5 is **domain/headroom-insufficient**
  for this diagnostic. Fail-closed codec (`narrow_accumulator_codec.py:419`)
  correctly raised at post-10-step ledger emit; treatment receipt never written.
- **W6 oracle trajectory** (per-step `global_max_abs_accumulator`): 4→8→12
  (≤step3), **16 at step4** (26/32 modules >15), 32/32 >15 by step6, **29 at
  step10** (0/32 >31; W6 [-31,31] held). Parent pt `9b4e311a` unchanged.
- **3-ledger (target vs actual):** forward 1.585 bpw unchanged; persistent
  oracle ≈8.0 bpw (q 2.0 + W6 6.0); treatment target ≈7.0 bpw (q 2.0 + W5
  5.0) — **not reached** (ledger emit blocked). NOT sub-2, NOT lossless.
- **Classifier:** receipt-level `R5_1_HARNESS_OR_LIVENESS_FAIL` (mechanically
  correct); science-level `R5_1_DOMAIN_OR_HEADROOM_FAIL`.
- **Honesty:** NO bank/`.pt`/sub-2/lossless/readiness/sub2_win.
  `R5_W5_BYTEPACKED_DECISION_PARITY_NOT_SUB2_STATEMENT`. W5 decision-parity
  not directly scored; ~50% max-range clip across all modules = strong indirect
  evidence against dense W5 on live geometry.

Claim boundary: W5 dense-width branch **CLOSED** for this lane. W6 =
practical dense-accumulator floor (corroborates prior R5 lossless null from the
domain angle). Do NOT run W4. Pivot = **sparse/structured vote representation
or changed vote law**, NOT narrower dense width. Clip-and-record observe-mode
optional future plan-dev packet only.

Run root:
`/home/gabe/claw-code-creditdir/transient_fp_credit/r5_1_w5_decision_parity_tier1_20260624_151005`.

Dual-accept: claude gate-1 `1782311411143` + co_lead gate-2 `1782311514192`;
board null synthesis `1782311588173`; co_lead concurrence `1782311605313`.

## Origin

Lane separated from `hrm-158.md` so the curriculum lane (90/90 bank gate,
full-density slices) stays distinct from the FP-free training-stack research
question (can sub-2-bit *persistent*-state ternary genuinely keep training).
The 3-ledger framing (forward 1.585 / persistent ~24 / export) and the
direction-flip-is-the-erosion-cause attribution are the load-bearing invariants
carried into the rule; their measurement receipts belong here.
