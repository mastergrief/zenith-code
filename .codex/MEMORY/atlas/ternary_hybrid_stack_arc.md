# Ternary-Hybrid Training Stack — receipts arc

Query-triggered receipts for `.codex/rules/ternary_hybrid_stack.md` (the
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

### 2026-06-24 — R5 offline falsification-screen → FIXED_ACC_REPRESENTATION_NULL (static_proxy)

Verdict: current fixed-accumulator representation/sparsity and this static
threshold/decay proxy do not yield a sub-2 persistent accumulator state.
(`FIXED_ACC_REPRESENTATION_NULL`, `static_proxy` only.)

Evidence (35-regime sweep over R5.1 W6-oracle sidecars, steps 3–10, 32
modules): sparse-hot min **36.56** bpw ≫ W6 6.0 reference; min domain
**4.86** bpw; min entropy **3.82** bits/lane; **0/35** `sub2_target_hit`
(target 2.0 bpw); `min_distance_from_2bpw` **2.865**. Crossing factual:
several regimes preserve crossing=1.0 while acc-term drops to ~5.3–5.7 bpw;
none approach sub-2; deeper threshold/decay cuts degrade crossing.

Explicitly NOT banked: "wide accumulator intrinsic to threshold-damped voting"
— needs votes-emitting dynamics-proof run. Honesty: `static_proxy`; applied-mask,
cap-order (`abs(new_acc)` ranking), and q-trajectory unscoreable (sidecars lack
per-lane votes/applied_indices).

Provenance: dual-accept `1782315588269` / `1782315687533`; run root
`r5_1_w5_decision_parity_tier1_20260624_151005`; artifacts
`d49ae333c58baa627eb54375c9f92fc9e254e6f3da19feb3b446511f721c6c5c` (json;
science-identical to the dual-accepted artifact; embedded `tool_source_sha256`
differs only due to the `:436` hygiene fix),
`10dd04872201bce0201b02c96d1358e73cd3e0e52d4316aef5aa347c42fe13ff` (csv).

### 2026-06-24 — q-ledger reframe + base-3 q-pack storage feasibility (Q_DENSE_PACK_LEDGER_WIN)

**Reframe (directional):** with the current banked 2-bit q-pack, strict sub-2
persistent state is blocked at the **q-ledger floor** even if accumulator
persistence is removed; the binding constraint is now **q-pack density**.
(`persistent_state_budget.py`: q `packed_2bit_ternary_reference` = 2.0 bpw,
target 2.0, `required_acc_physical ≤ 0`.)

**Q_DENSE_PACK_LEDGER_WIN:** existing base-3 5-trit/byte codec
(`q_entropy_packing.py:250` `pack_ternary_q_base3_5perbyte_reference`;
storage-only, **NOT checkpoint-wired**) gives inclusive q+scale
**1.600318 bpw < 2.0** on live 32-module / 29,360,128-lane surface; bit-exact
roundtrip **32/32**. 2-bit reference inclusive = **2.000314 bpw**. Acc headroom
≈ **0.40 bpw** at target 2.0 with acc omitted; metadata+scale ≈ **0.0003 bpw**
(not blocker). Parent `9b4e311a` read-only unchanged.

**Explicitly NOT banked:** not checkpoint-wired (2-bit seam remains
`trainer_sub2_authority.py:685/:790`; integration = separate gated build); not
full sub-2 persistent (acc non-persistence = separate gate); no
readiness/dynamics claim.

Provenance: reframe dual-accept `1782317459187` / `1782317569754`; q-pack win
dual-accept `1782318183500` / `1782318397500`; grounding receipt
`1782317965284-4964b8f6`.

### 2026-06-26 — V4-LIVE event-coded-drain envelope closure (JOINT_DRAIN_NOT_REACHABLE)

**Question:** can joint event+hot drain on the event-coded live carrier reach
R4v sub-2 (<2.0 bpw inclusive acc) at terminal Phase-A geometry?

**Verdict (bounded):** `JOINT_DRAIN_ENVELOPE_NOT_REACHABLE` **under available
parity evidence** → Path B `STRUCTURALLY_NOT_SUB2` as a **sub-2 mechanism at
terminal Phase-A geometry**. Event-coded-drain is **worse than dense int16** at
terminal (acc-term comparison below). This does **NOT** claim all
event-coded mechanisms or all accumulator alternatives are impossible — only
that this drain path cannot close the sub-2 gap on the measured terminal surface.

**Terminal measurements (run `2189e72004`, manifest-backed):**
- R4v inclusive acc: **85.10 bpw** vs dense int16 LIVE row **~24 bpw** (~3.5× worse)
- Events-floor: **53.78 bpw**; hot-floor: **31.32 bpw**; optimistic upper bound
  (full event clear + max parity-safe hot reduction): **31.32 bpw**
- Sub-2 acc budget: **7.34 MB**; optimistic total **~115 MB** → gap **15.66×**
- Residual: flip-to-2.0 needs **>93.6%** parity-safe hot reduction; flip-to-1.75
  needs **>94.4%**; synthetic band-sweep available fraction **0.0**; terminal
  hot-band parity at 115MB geometry **unmeasured** (numel=1024 sweep only)

**Harness / classification path:**
- V4-LIVE Phase-A diagnostic clean terminal (20/20 steps, parent `9b4e311a` unchanged)
- M2 cProfile → assert-path hash dedup (`9bcc008`) → R4v codec fix (`847704c`) →
  post-dedup rescreen (`2189e72004`) → joint-drain envelope CPU projector (`efcebf7`)
- Envelope verdict: `not_applicable` rollup transforms on manifest-only path;
  decisive bound = `optimistic_upper_bound`

**Artifacts (HRM repo `claw-code-hrm-text-158`):**
- Evidence manifest: `artifacts/consensus_prep/v4_live_phase_a_diagnostic_tier1_run_2189e72004_evidence_manifest.json`
- Envelope verdict: `artifacts/consensus_prep/v4_live_joint_drain_envelope_verdict_2189e72004.json` (sha `92b4cefe…`)
- Projector: `calm/hrm_text_158/native_full_stack/carrier_envelope_projector.py`

**Provenance (commits on HRM `feature/hrm-text-1.58`):**
- `fbabd34` — Phase 0 carrier-growth stub + scale-smoke
- `a6ec875` — Phase 1 production sidecar wire + terminal screen
- `efcebf7` — Phase 2 envelope projector + bankable verdict
- Parent checkpoint read-only: `9b4e311a`

**Next (explicitly NOT closed here):** dense-acc-width investigation / int16
vote-acc dominator alternatives (co_lead refinement `1782491192366`).

### 2026-06-28 — Arc #2b D-recompute window horizon sizing + B1 H=200 de-censor (envelope-only)

**Chain (HRM repo `claw-code-hrm-text-158`, `feature/hrm-text-1.58`):**
- H=100 worst-case run (`2189e72015`): right-censored INCONCLUSIVE at sizing horizon
  H=100 (`growth_branch=RIGHT_CENSORED_LOWER_BOUND`, kworst_weighted=100).
- STEP-1 classifier↔packet input-manifest bind (`0357e8a`).
- STEP-1b run_root-local-log preference + fail-closed guards (`7f77640`).
- Slice-A distributional p99 `quantile_acc_sizing` (`e48a5bb`):
  `D_RECOMPUTE_QUANTILE_SUB2_CANDIDATE`, envelope ~0.000278 bpw; faithful-cap caveat:
  99.8%-deferred so it sizes the APPLIED persistent footprint, NOT worst-case.
- B1 H=200 de-censor wiring (`556623e`): `GROWTH_DECENSORED_SIZED_AT_HORIZON`,
  packet-driven `horizon_ladder=[25,50,100,200]`, `sizing_horizon_h=200`.

**Liveness-failure incident (run `2189e72016`, code HEAD `556623e`):**
- Died at step 73/200 — CPU/materialization liveness failure (step_update hang >300s,
  fail-closed faulthandler; traceback lost to mirrored-stderr fileno bypass).
- Root cause: `emit.py:503` full-population `vote_lane_values` `.item()` loop
  (O(numel)/key/step) built unconditionally then discarded under
  `STRESS_TAIL_POLICY_HORIZON_FIXED` (horizon-fixed selector reads manifest lanes only;
  `selector.py:477-485`).
- Fix = Slice-1 (`5cc8fb9`): emit.py horizon-fixed lane-first skip + survivability
  S1-S4 (stderr/faulthandler tee to run_root, phase stack dump on breach,
  faulthandler on run.log fd).
- Relaunch packet `2189e72017` + ancestor-checked generic driver (`95340d3`).
- Dead `2189e72016` archived (`mv`, never rm) as
  `..._2189e72016_DEAD_liveness_step73` forensic record.

**B1 verdict (run `2189e72017`, dual-accepted gate-1 `1782679725046` + co_lead
gate-2 `1782679899343`) — EXACT language, no upgrade:**

B1 de-censor succeeded at H=200 (worst-case K* no longer right-censored;
kworst_weighted=180<200, right_censor_rate=0.0, parity_fail_count=0,
gapped_lane_count=0; k99=112, k95=39). The envelope model sizes the worst-case acc
footprint inclusive_acc_bpw=0.00064237 at window_k=180, strictly under the
effective_acc_budget_bpw=0.4. BUT in-vivo validation is
INCONCLUSIVE_REAL_DENSITY_EXCEEDS_ENVELOPE (peak_backlog_depth=130816,
total_global_rate_cap_deferred=26,162,688 vs accepted=51,712, total_flip_events=1) →
final_sizing_verdict=SIZED_WINDOW_ONLY_NOT_SUB2,
final_verdict_scope=envelope_model_only, recommended_law_eligible=false,
primary_classifier=D_RECOMPUTE_SIZED_NOT_SUB2, requires_slice5_live_validation=true.
quantile_sub2_candidate=false (growth_branch_not_right_censored_lower_bound). NOT
sub-2, NOT recommended-law, NOT in-vivo-bound, NOT physical-persistent-sub-2.

**Liveness fix validated on relaunch:** step 73=11.45s; 200/200 clean @~13s/step;
no breach (`last_active_phase.json` liveness_failure=false, guard_event=cleared).

**Run integrity:** bind PASS spec_sha `4f368336`, log run_root-local, jsonl 3600
lines, parent .pt `9b4e311a` read-only match ×4.

**Instrumentation nit (Slice-2 cleanup, not science):** the armed-guard pre-registered
dump writes `failure_class="LIVENESS_FAILURE"` on `guard_event="enter"` — misleading
nomenclature (distinguish armed vs fired).

**Next rung:** Slice-5 in-vivo live validation (route to a real, non-envelope
worst-case sub-2 claim).

## Origin

Lane separated from `hrm-158.md` so the curriculum lane (90/90 bank gate,
full-density slices) stays distinct from the FP-free training-stack research
question (can sub-2-bit *persistent*-state ternary genuinely keep training).
The 3-ledger framing (forward 1.585 / persistent ~24 / export) and the
direction-flip-is-the-erosion-cause attribution are the load-bearing invariants
carried into the rule; their measurement receipts belong here.
