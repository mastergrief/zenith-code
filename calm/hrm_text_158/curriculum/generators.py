"""Phase 3 curriculum synthetic generators.

Per codex msg 1779458774209 + 1779459000384 (2-axis curriculum locked):

Axis 1 — math complexity under stable language wrapper:
  R0: digit copy / format        `what is N?` -> `N`
  R1: identity-operator bridge   `what is A plus 0?` -> `A`   (also `0 plus A`, `A minus 0`)
                                 [redefined per codex msg 1779466025267 after R1 v1
                                  failed acquisition; bridges R0 copy -> two-operand parsing
                                  + operator binding without full arithmetic]
  R1b1: single-template +1       `what is A plus 1?` -> `A+1`  (A in [1,98])
                                 [codex msg 1779469364293 + 1779469638068 after R1b v2
                                  failed at 0.845 with 2x training steps (0d152dd "more-
                                  updates hypothesis falsified"). Falsifier-protocol split:
                                  isolate the simplest sub-skill (`A plus K` position) so
                                  the model can acquire it cleanly before symmetric +1 and
                                  -1 variations land in successor rungs.
                                  PASSED at 0.930 exact (66b9747).]
  R1b2a: low-A subtraction       `what is A minus 1?` -> `A-1` (A in [1,19])
                                 [codex msg 1779472124507 + 1779472300306 after R1b2
                                  FAIL at 0.860 + R1b1 retention -0.050 decay
                                  (6fd2fec). Isolates minimal `-1` operator on
                                  low-A operands (no two-digit borrow) with stronger
                                  replay (ratio 0.50) to anchor R1b1; falsifier on
                                  whether operator coexistence can be controlled.]
  R1b2: single-template -1       `what is A minus 1?` -> `A-1` (A in [1,99])
                                 [codex msg 1779471073874 + 1779471212090 after R1b1
                                  PASS. Isolated subtraction at the same operand
                                  position as R1b1; FAILED at 0.860 G1 with R1b1
                                  retention -0.050 decay (6fd2fec). Now DIAGNOSIS-
                                  ONLY pending R1b2a outcome; A_minus_1 over [1,99]
                                  overlaps R1b2a's [1,19] subset by construction.]
  R1b: minimal arithmetic (±1)   `what is A plus 1?` -> `A+1` (also `1 plus A`, `A minus 1`)
                                 [DIAGNOSIS-ONLY after R1b v2 failure (codex msg
                                  1779469638068). Stays in RUNG_NAMES for backward-
                                  compat; out of build_rung_splits default tuple because
                                  its A_plus_1 rows overlap R1b1's by construction.
                                  Successor design tbd post-R1b1.]
  R1b3: constant K=2 addition    `what is A plus 2?` -> `A+2`  (A in [1,97])
                                 [codex msg 1779479973262-6d7445d2 after R2a v1 failed 0.045
                                  at 558fcc1; variable-B reframed as structural blocker.
                                  Extends locked constant-B single-template pattern (R1b1
                                  K=1, R1b2 K=-1) to K=2 BEFORE attempting variable-B.
                                  Output [3,99]; bucket-stratified 80/20 = 77 train + 20 held.
                                  PASSED at 0.930 via v2 schedule (lr 5e-4 / 1500 steps) at
                                  175d327; new default schedule established.]
  R1b4: constant K=3 addition    `what is A plus 3?` -> `A+3`  (A in [1,96])
                                 [codex msg 1779482125661-b2c0ca2a after R1b3 v2 PASS at
                                  175d327. Continues locked constant-K pattern (K=1, K=-1,
                                  K=2 all PASSED) to K=3 before grouped K or variable-B.
                                  Output [4,99]; bucket-stratified 80/20 = 76 train + 20 held.
                                  v1 FAILED at 7b53368: standard 0.885 below G1 0.90 due to
                                  one_digit thin-pool (2 heldout rows sampled ~22× each via
                                  rng.choice). DIAGNOSIS-ONLY after v1 fail per codex msg
                                  1779483673737-20ff22ab; R1b4v2 (one_digit-exhaustive) is
                                  the active-chain successor.]
  R1b4v2: K=3 addition, one-digit-exhaustive partition (codex msg 1779483673737-20ff22ab
                                  after R1b4 v1 fail at 7b53368). Same question/output as
                                  R1b4 (`what is A plus 3?` -> A+3) but measurement/support
                                  redesign: one_digit A=1..9 EXHAUSTIVE in train (all 9 rows);
                                  two_digit A=10..96 80/20 stratified (heldout = 18 two_digit
                                  rows only, zero one_digit). Separate deterministic 9-row
                                  one_digit exhaustive audit served via
                                  `r1b4v2_one_digit_audit_rows`. Preserves R1b4 immutable as
                                  failed diagnostic. ADVANCED via seed=2 head at b368b81.]
  R1b5: constant K=4 addition    `what is A plus 4?` -> `A+4`  (A in [1, 95])
                                 [codex msgs 1779488238721-49f03cc9 +1 advance R1b4v2 +
                                  R1b5 direction; 1779523412979-ff88b885 +1 design+implement
                                  R1b5 with keyed per-rung audits. Continues constant-K
                                  jigsaw (K=1, K=-1, K=2, K=3 all passed). Bakes in R1b4v2
                                  lessons FROM START:
                                    - one_digit A=1..9 exhaustive in train (no thin pool)
                                    - two_digit A=10..95 carry-stratified 80/20 (NEW
                                      dimension): carry-bucket {units in {6,7,8,9}} = 32 vals
                                      -> 25 train + 7 held; non-carry {units in {0..5}} = 54
                                      vals -> 43 train + 11 held. TOTAL: 86 -> 68 train + 18
                                      held with guaranteed 7 carry + 11 non-carry in held.
                                  Output [5, 99]; no 3-digit class. Audit: 9-row exhaustive
                                  one_digit via r1b5_one_digit_audit_rows. ADVANCED via
                                  seed=17 head. Parent ckpt:
                                  hrm_text_158_phase3_R1b4v2_seed0002_final.pt.]
  R1b6: constant K=5 addition    `what is A plus 5?` -> `A+5`  (A in [1, 94])
                                 [codex msg 1779545956176-4a8cfc3e +1 K=5 naming +
                                  1779545998312-88256068 extra prior_rungs receipt; user
                                  provenance via codex msg 1779545575582-7c52a912 of gabe
                                  verbatim "ok implement, full prov". Continues constant-K
                                  jigsaw (K=1, K=-1, K=2, K=3, K=4 all PASSED). Mirrors
                                  R1b5 carry-stratified design at K=5:
                                    - one_digit A=1..9 exhaustive in train (no held)
                                    - two_digit A=10..94 carry-stratified 80/20: carry
                                      {units in {5,6,7,8,9}} = 40 vals -> 32 train + 8 held;
                                      non-carry {units in {0..4}} = 45 vals -> 36 train + 9
                                      held. TOTAL: 85 -> 68 train + 17 held (guaranteed 8
                                      carry + 9 non-carry); with one_digit 94 -> 77 + 17.
                                  Output [6, 99]; no 3-digit class. Audit: 9-row exhaustive
                                  one_digit via r1b6_one_digit_audit_rows. ADVANCED via
                                  replay50_lr5e4 head at 128b097 (R1b6 50/50 + audit 9/9 +
                                  all priors 50/50 + all keyed audits 9/9). Parent ckpt:
                                  hrm_text_158_phase3_R1b5_seed0017_final.pt. Faststack
                                  enabled: --use-ternary-bulk + --use-native-ternary-train
                                  training; --use-cached-ternary-infer + --use-kv-cache-decode
                                  + --use-batched-probe-eval probe.]
  R1b7: constant K=6 addition    `what is A plus 6?` -> `A+6`  (A in [1, 93])
                                 [codex msg 1779547753761-5711d790 +1 K=6 after R1b6 commit
                                  128b097; durable gabe provenance relay 1779547541812 of
                                  gabe verbatim "you have full provenance until im back".
                                  Continues constant-K jigsaw (K=1, K=-1, K=2, K=3, K=4,
                                  K=5 all PASSED). Mirrors R1b5/R1b6 carry-stratified
                                  design at K=6:
                                    - one_digit A=1..9 exhaustive in train (no held)
                                    - two_digit A=10..93 carry-stratified 80/20: carry
                                      {units in {4,5,6,7,8,9}} = 48 vals -> 38 train + 10
                                      held; non-carry {units in {0..3}} = 36 vals -> 28
                                      train + 8 held. TOTAL: 84 -> 66 train + 18 held
                                      (guaranteed 10 carry + 8 non-carry); with one_digit
                                      93 -> 75 train + 18 held.
                                  Output [7, 99]; no 3-digit class. Audit: 9-row exhaustive
                                  one_digit via r1b7_one_digit_audit_rows. Parent ckpt
                                  (accepted candidate from R1b2-repair commit 9c8f800
                                  after R1b6 candidate full-chain baseline revealed
                                  R1b2=0.78 pre-existing gap; R1b2-repair lifted R1b2 to
                                  0.92 + held all priors >=parent + all keyed audits 9/9):
                                  hrm_text_158_phase3_R1b2_repair_seed0017_replay50_lr5e4_final.pt.
                                  Faststack carried forward from R1b6 winning combo.
                                  Full-chain probe required per codex msg 1779547753761:
                                  R0..R1b7 with absolute + parent-relative deltas + keyed
                                  audit deltas; **R1b2 explicit hard retention gate** per
                                  codex msg 1779549330637 (held 0.92->0.92 in this run).]
  R1b8: constant K=7 addition    `what is A plus 7?` -> `A+7`  (A in [1, 92])
                                 [codex msg 1779550489408-f40f66ab +1 K=7 after R1b7 commit
                                  682659b ADVANCED via R1b2-retained chain + A0 exhaustive
                                  audit 1071/1072 = 99.91% PASS. Continues constant-K
                                  jigsaw (K=1, K=-1, K=2, K=3, K=4, K=5, K=6 all PASSED).
                                  Mirrors R1b5/R1b6/R1b7 carry-stratified design at K=7:
                                    - one_digit A=1..9 exhaustive in train
                                    - two_digit A=10..92 carry-stratified 80/20: carry
                                      {units in {3..9}} = 56 vals -> 44 train + 12 held;
                                      non-carry {units in {0..2}} = 27 vals -> 21 train +
                                      6 held. TOTAL: 83 -> 65 train + 18 held; with
                                      one_digit 92 -> 74 train + 18 held.
                                  Output [8, 99]; no 3-digit class. Audit: 9-row exhaustive
                                  one_digit via r1b8_one_digit_audit_rows. Parent ckpt
                                  (accepted R1b7 candidate from commit 682659b):
                                  hrm_text_158_phase3_R1b7_seed0017_replay50_lr5e4_from_R1b2_repair_final.pt.
                                  Explicit R1b2 boundary watch per codex 1779550489408:
                                  verify `what is 10 minus 1?` decodes correctly post-train;
                                  R1b2 full-support rate should stay >=0.99 preferred.]
  R2a: teens addition-only       `what is 13 plus 7?` -> `20` (A in [10,19], B in [2,9])
                                 [DIAGNOSIS-ONLY after v1 failed 0.045 at 558fcc1; variable-B
                                  is the blocker, not operator mixing. R1b3 is the active
                                  successor (constant K=2 first).]
  R2: teens variable-B ±         `what is 13 plus 7?` -> `20`   (A in [10,19], B in [2,9])
                                 [DIAGNOSIS-ONLY after v1+v2 fail at c2f4f8d; A_plus_B rows
                                  overlap R2a by construction. Reachable via explicit rungs=
                                  for diagnosis; auto-excluded from positional via
                                  DIAGNOSIS_ONLY_RUNGS in replay.py.]
  R3: multiplication only        `what is A times B?` -> `C`  (NO division)
  R4: multi-step compound        `what is A plus B times C?` -> `D`

Axis 2 — language variability grows, math stable from R0-R4:
  R5: paraphrases over R0-R4 primitives
  R6: templated word problems
  R7: GSM8k full corpus (delegated to existing load_gsm8k_splits)

All generators are deterministic per (rung, seed, split). split="train" and
split="held_out" produce NON-OVERLAPPING ROWS via either disjoint operand
ranges (R2/R4) or deterministic row partitioning over a shared support
(R0/R1/R1b/R3 enumerate / stratified partitions). Cross-rung train/held_out
invariant enforced by splits.assert_no_train_holdout_overlap.

Win/falsifier per codex msg 1779457170889 + 1779458774209:
- monotonic rung accuracy
- low forgetting (>90% retention on prior rungs)
- canonical 17×23 tracks multiplication-rung mastery
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, Literal


RUNG_NAMES = ("R0", "R1", "R1b1", "R1b2a", "R1b2", "R1b3", "R1b4", "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b", "R2a", "R2", "R3", "R4", "R5", "R6", "R7")


def _stable_seed(*parts) -> int:
    """Process-stable seed derivation.

    Python's builtin `hash()` is salted per-process under default
    `PYTHONHASHSEED=random`, so `hash(("R1_partition", 42))` returns
    different values across restarts — that would silently produce
    different train/held_out partitions per run, invalidating
    retention probes and ckpt comparability.

    `hashlib.sha256(repr(parts).encode("utf-8"))` is deterministic
    across processes and Python versions for the same input tuple.
    """
    blob = repr(parts).encode("utf-8")
    digest = hashlib.sha256(blob).digest()[:4]
    return int.from_bytes(digest, "little")


# Per-rung sampling spec.
#
# CROSS-RUNG INVARIANT (per codex msg 1779458774209): NO row in any rung's
# held_out may appear in ANY rung's train. Operand ranges chosen to make
# this hold by construction:
#
#   - R0 template "what is N?" is UNIQUE to R0 (no other rung uses it).
#     R0 uses STRATIFIED in-distribution partition over [0,99]
#     (codex msg 1779464341737 after v1 OOD-shift fix): bucket [0,9] and
#     bucket [10,99] each partitioned 80/20 separately so train + held_out
#     both contain 1- and 2-digit Ns. No operand >= 100 in either split.
#
#   - R1 (identity-bridge per codex msg 1779466025267): three templates
#     `A plus 0` / `0 plus A` / `A minus 0`, all output A. A in [0,99]
#     stratified by (template, digit-bucket) 80/20. R2 cannot collide:
#     R2 train requires B in [10,99] so no R2 row has B=0; R1 always
#     has B=0 or A=0 in the question text. Disjoint by row content.
#
#   - R1b (minimal ±1 bridge per codex msg 1779467425298): three templates
#     `A plus 1` / `1 plus A` / `A minus 1` with per-template A ranges
#     chosen to drop the cross-rung row collisions:
#         A_plus_1   A in [1,98]  -- drop A=0 (else "what is 0 plus 1?" -> 1
#                                   duplicates R1 0_plus_A A=1)
#                                  -- cap A=98 (keeps output <= 99, no new
#                                     digit-length class)
#         1_plus_A   A in [2,98]  -- drop A=0 (else "what is 1 plus 0?" -> 1
#                                   duplicates R1 A_plus_0 A=1)
#                                  -- drop A=1 (else "what is 1 plus 1?" -> 2
#                                   duplicates intra-R1b A_plus_1 A=1)
#         A_minus_1  A in [1,99]  -- drop A=0 (output would be -1, schema
#                                   mismatch)
#     Output stays in [0,99]. R2 cannot collide: R2 train requires
#     B in [10,99] so no R2 row has B=1. Pool 234 train + 60 held_out.
#
#   - R3 template "what is A times B?" is UNIQUE to R3 (no other rung
#     uses "times" with this shape). 17×23 canonical FORCED in held_out.
#
#   - R4 template "what is A op1 B op2 C?" is UNIQUE (3-operand structure).
#
#   - R5 uses arithmetic-operator chars (+, -, *) instead of words; R6
#     uses word-problem templates. Both have unique surface forms.
#
# Implementation: R0 + R1 + R1b + R3 enumerate full operand spaces +
# deterministic shuffle-partition (stratified by template / digit-bucket).
# R2 / R4 use disjoint operand ranges (no overlap by construction).
#
# R0 design correction (codex msg 1779464341737-43a42cae after R0 launch
# msg 1779464300667 reported G1 fail):
#   - Previous R0 design: train [0,99] vs held_out [100,999]. Tests OOD
#     length generalization, not in-distribution digit-copy memorization
#     (model learned 1-2 digit copy but couldn't extend to 3-digit -> G1=0).
#   - New R0 design: STRATIFIED in-distribution partition over [0,99].
#     Train + held_out both contain 1-digit AND 2-digit examples. Bucket
#     [0,9] partitioned 80/20 separately from [10,99] so the digit-length
#     mix is preserved on both sides. Tests in-distribution memorization
#     of digit copy (the actual R0 primitive).
#   - R0 max N <= 99 in both splits. OOD length generalization is NOT
#     an R0 gate.

_RUNG_SPEC: dict[str, dict[str, dict]] = {
    "R0": {
        # R0 stratified partition: bucket [0,9] (one-digit) + bucket [10,99]
        # (two-digit) split 80/20 each, then unioned per split. Held_out
        # max N <= 99 — in-distribution memorization gate.
        "train":     {"N_range": (0, 99), "partition": "enumerate_stratified"},
        "held_out":  {"N_range": (0, 99), "partition": "enumerate_stratified"},
    },
    # R1 (codex msg 1779466025267 redefinition): identity-bridge.
    # 3 templates (A_plus_0 / 0_plus_A / A_minus_0) × [0,99] A values,
    # stratified by (template, digit-bucket) 80/20.
    # Train 239 (template, A) pairs; held_out 60. Both splits contain
    # all 3 templates AND both 1-digit + 2-digit A. "0_plus_A" drops A=0
    # (collision with "A_plus_0" + A=0, both emit "what is 0 plus 0?").
    "R1": {
        "train":     {"A_range": (0, 99), "partition": "enumerate_stratified_identity"},
        "held_out":  {"A_range": (0, 99), "partition": "enumerate_stratified_identity"},
    },
    # R1b1 (codex msg 1779469364293 + 1779469638068 falsifier-protocol split
    # after R1b v2 failed at 0.845 with 2x steps): SINGLE-template `A plus 1`
    # over A in [1,98], bucket-stratified. Drops A=0 (would emit "what is 0
    # plus 1?" -> 1 -- duplicates R1 0_plus_A with A=1) and A=99 (output
    # would saturate at 100, introducing new digit-length class). Inserted
    # BEFORE R1b in RUNG_NAMES so trainer's prior_rungs derivation
    # (RUNG_NAMES[:cur_idx]) auto-resolves to (R0, R1) and the failed full
    # R1b is excluded from the active chain.
    "R1b1": {
        "train":     {"A_range": (1, 98), "partition": "enumerate_stratified_r1b1"},
        "held_out":  {"A_range": (1, 98), "partition": "enumerate_stratified_r1b1"},
    },
    # R1b2a (codex msg 1779472124507 + 1779472300306 after R1b2 FAILED
    # at 0.860 G1 + R1b1 retention -0.050 decay, 6fd2fec): SINGLE-template
    # `A minus 1` over A in [1,19] (one-digit + teens, no two-digit borrow),
    # bucket-stratified. Designed to isolate the minimal `-1` operator on
    # low-A operands; pairs with replay_ratio=0.50 (vs prior 0.30) to
    # anchor R1b1 retention stronger. Output [0,18]. Inserted BEFORE
    # diagnosis-only R1b2/R1b in RUNG_NAMES so trainer prior_rungs auto-
    # resolves to (R0, R1, R1b1).
    "R1b2a": {
        "train":     {"A_range": (1, 19), "partition": "enumerate_stratified_r1b2a"},
        "held_out":  {"A_range": (1, 19), "partition": "enumerate_stratified_r1b2a"},
    },
    # R1b2 (codex msg 1779471073874 + 1779471212090 after R1b1 PASS at
    # 66b9747): SINGLE-template `A minus 1` over A in [1,99],
    # bucket-stratified. Drops A=0 (output would be -1; negative output
    # mismatches schema). Inserted BEFORE diagnosis-only R1b in
    # RUNG_NAMES so trainer's prior_rungs derivation
    # (RUNG_NAMES[:cur_idx]) auto-resolves to (R0, R1, R1b1).
    "R1b2": {
        "train":     {"A_range": (1, 99), "partition": "enumerate_stratified_r1b2"},
        "held_out":  {"A_range": (1, 99), "partition": "enumerate_stratified_r1b2"},
    },
    # R1b (codex msg 1779467425298): minimal arithmetic bridge from R1
    # identity to single-digit ±1. Templates A_plus_1 / 1_plus_A /
    # A_minus_1 with A-range constrained per-template to prevent
    # cross-template AND cross-rung row collisions:
    #   A_plus_1:  A in [1,98]  (drop A=0 -> avoids R1 0_plus_A/A=1 collision)
    #   1_plus_A:  A in [2,98]  (drop A=0 -> avoids R1 A_plus_0/A=1 collision;
    #                             drop A=1 -> avoids intra-R1b A_plus_1/A=1 collision)
    #   A_minus_1: A in [1,99]  (drop A=0 to avoid negative output)
    # Output stays in [0,99] -> no new digit-length class.
    #
    # NOTE (codex msg 1779469638068): R1b is now diagnosis-only. After R1b
    # v2 failed at 0.845 ("more-updates" hypothesis falsified, 0d152dd),
    # falsifier-protocol split to R1b1 (single template). R1b's A_plus_1
    # rows overlap R1b1's A_plus_1 rows by construction (same template +
    # same A range [1,98]) so R1b is NOT in the active chain and NOT in
    # build_rung_splits default tuple. R1b's own generator + tests stay
    # green as diagnosis-only and as the R1b1 successor design parent.
    "R1b": {
        "train":     {"A_range": (1, 99), "partition": "enumerate_stratified_pm1"},
        "held_out":  {"A_range": (1, 99), "partition": "enumerate_stratified_pm1"},
    },
    # R1b3 (codex msg 1779479973262-6d7445d2 after R2a v1 failed 0.045 at
    # 558fcc1): constant K=2 addition. Extends locked constant-B single-
    # template pattern (R1b1 K=1, R1b2 K=-1) to K=2 BEFORE variable-B.
    #   A in [1, 97], template `A_plus_2`. Output [3, 99] (no 3-digit).
    # Stratified by digit bucket per R1b1 pattern:
    #   one_digit [1, 9]:    9 vals  -> 7 train + 2 held
    #   two_digit [10, 97]: 88 vals  -> 70 train + 18 held
    #   TOTAL:              97 vals  -> 77 train + 20 held_out
    # B=2 disjoint from R1 (B=0), R1b1 (B=1), R1b2 (B=1), R2a (variable B
    # in [2,9] but R2a is diagnosis-only; A range [10,19] overlaps R1b3
    # for A in [10,19] B=2 -> stays excluded from active chain).
    "R1b3": {
        "train":     {"A_range": (1, 97), "partition": "enumerate_stratified_r1b3"},
        "held_out":  {"A_range": (1, 97), "partition": "enumerate_stratified_r1b3"},
    },
    # R1b4 (codex msg 1779482125661-b2c0ca2a after R1b3 v2 schedule PASS
    # at 175d327): constant K=3 addition. Continues locked constant-B
    # single-template pattern (R1b1 K=1, R1b2 K=-1, R1b3 K=2) to K=3
    # before considering grouped constant-K or variable-B.
    #   A in [1, 96], template `A_plus_3`. Output [4, 99] (no 3-digit).
    # Stratified by digit bucket per R1b1/R1b3 pattern:
    #   one_digit [1, 9]:    9 vals -> 7 train + 2 held
    #   two_digit [10, 96]: 87 vals -> 69 train + 18 held
    #   TOTAL:              96 vals -> 76 train + 20 held_out
    "R1b4": {
        "train":     {"A_range": (1, 96), "partition": "enumerate_stratified_r1b4"},
        "held_out":  {"A_range": (1, 96), "partition": "enumerate_stratified_r1b4"},
    },
    # R1b4v2 (codex msg 1779483673737-20ff22ab after R1b4 v1 fail at 7b53368):
    # one-digit-exhaustive partition redesign. PROVENANCE-PRESERVING new
    # rung — R1b4 stays immutable as v1 failed diagnostic. Same question /
    # output (`what is A plus 3?` -> A+3) but measurement support design
    # fixed: one_digit support is too small (9 vals) to split 80/20 into
    # train (7) + heldout (2) and have `probe_curriculum`'s eval_cap=200
    # `rng.choice` sampling produce robust generalization signal — the
    # 2-row heldout one_digit bucket gets sampled ~22x each on average.
    #
    # Partition:
    #   one_digit [1, 9]:    9 vals -> 9 EXHAUSTIVE train + 0 held_out
    #   two_digit [10, 96]: 87 vals -> 69 train + 18 held_out (80/20)
    #   TOTAL:              96 vals -> 78 train + 18 held_out integers A
    # Held_out contains ONLY two_digit rows; one_digit mastery is gated
    # via a separate deterministic 9-row exhaustive audit served by
    # `r1b4v2_one_digit_audit_rows(seed)`.
    #
    # B=3 stays disjoint from priors R0/R1/R1b1/R1b2/R1b3 same as R1b4.
    "R1b4v2": {
        "train":     {"A_range": (1, 96), "partition": "enumerate_stratified_r1b4v2"},
        "held_out":  {"A_range": (10, 96), "partition": "enumerate_stratified_r1b4v2"},
    },
    # R1b5 (codex msgs 1779488238721-49f03cc9 + 1779523412979-ff88b885 after
    # R1b4v2 advance at b368b81): constant K=4 addition. Continues locked
    # constant-K single-template jigsaw (K=1, K=-1, K=2, K=3 all passed).
    # Bakes in R1b4v2 measurement/support lessons FROM START:
    #   one_digit A=1..9: all 9 to TRAIN exhaustive (NO thin-pool heldout)
    #   two_digit A=10..95 carry-stratified 80/20 (NEW dimension):
    #     carry (units+4>=10, units in {6,7,8,9}): 32 vals -> 25 train + 7 held
    #     non-carry (units in {0..5}):              54 vals -> 43 train + 11 held
    #     TOTAL: 86 -> 68 train + 18 held (guaranteed 7 carry + 11 non-carry)
    # Output [5, 99]; no 3-digit class. B=4 disjoint from active priors
    # (R0 has no B; R1 B=0; R1b1 B=1; R1b2 B=1 minus; R1b3 B=2; R1b4v2 B=3).
    # Diagnosis-only R2/R2a have B in [2,9] which overlaps B=4 historically;
    # cross-rung invariant covers active-chain (R2/R2a not in default).
    "R1b5": {
        "train":     {"A_range": (1, 95), "partition": "enumerate_stratified_r1b5"},
        "held_out":  {"A_range": (10, 95), "partition": "enumerate_stratified_r1b5"},
    },
    # R1b6 (codex msg 1779545956176-4a8cfc3e +1 K=5 naming + 1779545998312-88256068
    # extra prior_rungs receipt; user provenance via codex msg 1779545575582-7c52a912
    # of gabe verbatim "ok implement, full prov"): constant K=5 addition. Continues
    # locked constant-K jigsaw (K=1, K=-1, K=2, K=3, K=4 all PASSED). Mirrors R1b5
    # carry-stratified design:
    #   one_digit A=1..9: 9 vals -> 9 train exhaustive (NO held_out per R1b4v2 lesson)
    #   two_digit A=10..94 carry-stratified 80/20 (carry now units in {5..9} since K=5):
    #     carry (units+5>=10, units in {5,6,7,8,9}): 40 vals -> 32 train + 8 held
    #     non-carry (units in {0..4}):               45 vals -> 36 train + 9 held
    #     TOTAL: 85 -> 68 train + 17 held (guaranteed 8 carry + 9 non-carry)
    #   TOTAL WITH ONE_DIGIT: 94 vals -> 77 train + 17 held_out
    # Output [6, 99]; no 3-digit class (max A+5 = 94+5 = 99). B=5 disjoint from
    # active priors (R0 has no B; R1 B=0; R1b1 B=1; R1b2 B=1 minus; R1b3 B=2;
    # R1b4v2 B=3; R1b5 B=4). Audit: 9-row exhaustive one_digit via
    # r1b6_one_digit_audit_rows. Parent ckpt:
    # hrm_text_158_phase3_R1b5_seed0017_final.pt.
    "R1b6": {
        "train":     {"A_range": (1, 94), "partition": "enumerate_stratified_r1b6"},
        "held_out":  {"A_range": (10, 94), "partition": "enumerate_stratified_r1b6"},
    },
    # R1b7 (codex msg 1779547753761-5711d790 +1 K=6 originally after R1b6
    # commit 128b097; codex msg 1779549330637-876c3453 +1 rebase onto
    # R1b2-repair commit 9c8f800 after full-chain baseline revealed
    # R1b2=0.78 pre-existing gap then R1b2-repair lifted to 0.92; durable
    # gabe provenance relay 1779547541812-46e20177 "you have full
    # provenance until im back"): constant K=6 addition. Continues locked constant-K jigsaw
    # (K=1, K=-1, K=2, K=3, K=4, K=5 all PASSED). Same shape as R1b5/R1b6:
    #   one_digit A=1..9: 9 vals -> 9 train exhaustive
    #   two_digit A=10..93 carry-stratified 80/20 (carry-units {4..9} since K=6):
    #     carry (units+6>=10, units in {4,5,6,7,8,9}): 48 vals -> 38 train + 10 held
    #     non-carry (units in {0..3}):                36 vals -> 28 train +  8 held
    #     TOTAL: 84 -> 66 train + 18 held (guaranteed 10 carry + 8 non-carry)
    #   TOTAL WITH ONE_DIGIT: 93 vals -> 75 train + 18 held_out
    # Output [7, 99]; no 3-digit class (max A+6 = 93+6 = 99). B=6 disjoint
    # from active priors (R0 has no B; R1 B=0; R1b1 B=1; R1b2 B=1 minus;
    # R1b3 B=2; R1b4v2 B=3; R1b5 B=4; R1b6 B=5). Audit: 9-row exhaustive
    # one_digit via r1b7_one_digit_audit_rows. Parent ckpt (accepted
    # candidate from R1b2-repair commit 9c8f800, which fixed the
    # R1b2=0.78 pre-existing gap revealed by R1b6 candidate full-chain
    # baseline -- now R1b2=0.92, all other priors >=parent, all keyed
    # audits 9/9):
    # hrm_text_158_phase3_R1b2_repair_seed0017_replay50_lr5e4_final.pt.
    # Per codex msg 1779549330637-876c3453 +1 R1b7 rebase: R1b2 is
    # explicit hard retention gate for R1b7 acceptance (held 0.92->0.92
    # in this run).
    "R1b7": {
        "train":     {"A_range": (1, 93), "partition": "enumerate_stratified_r1b7"},
        "held_out":  {"A_range": (10, 93), "partition": "enumerate_stratified_r1b7"},
    },
    # R1b8 (codex msg 1779550489408-f40f66ab +1 K=7 after R1b7 commit 682659b
    # ADVANCED via R1b2-retained chain + A0 exhaustive audit 1071/1072 PASS).
    # Continues locked constant-K jigsaw (K=1, K=-1, K=2, K=3, K=4, K=5, K=6
    # all PASSED). Mirrors R1b5/R1b6/R1b7 carry-stratified design at K=7:
    #   one_digit A=1..9: 9 vals -> 9 train exhaustive
    #   two_digit A=10..92 carry-stratified 80/20 (carry-units {3..9} since K=7):
    #     carry (units+7>=10, units in {3,4,5,6,7,8,9}): 56 vals -> 44 train + 12 held
    #     non-carry (units in {0,1,2}):                  27 vals -> 21 train +  6 held
    #     TOTAL: 83 -> 65 train + 18 held (guaranteed 12 carry + 6 non-carry)
    #   TOTAL WITH ONE_DIGIT: 92 vals -> 74 train + 18 held_out
    # Output [8, 99]; no 3-digit class (max A+7 = 92+7 = 99). B=7 disjoint
    # from active priors (R0 no B; R1 B=0; R1b1 B=1; R1b2 B=1 minus; R1b3
    # B=2; R1b4v2 B=3; R1b5 B=4; R1b6 B=5; R1b7 B=6). Audit: 9-row exhaustive
    # one_digit via r1b8_one_digit_audit_rows. Parent ckpt (accepted R1b7
    # candidate from commit 682659b, which advanced the math chain through
    # K=6 with R1b2 hard-retention preserved):
    # hrm_text_158_phase3_R1b7_seed0017_replay50_lr5e4_from_R1b2_repair_final.pt.
    # Per codex msg 1779550489408 R1b8 launch +1: R1b2 boundary watch
    # explicit (verify `what is 10 minus 1?` decodes correctly post-train;
    # R1b2 full-support rate should stay >=0.99 preferred, hard floor 0.90).
    "R1b8": {
        "train":     {"A_range": (1, 92), "partition": "enumerate_stratified_r1b8"},
        "held_out":  {"A_range": (10, 92), "partition": "enumerate_stratified_r1b8"},
    },
    # R2a (codex msg 1779478819906-0e30503e after full R2 failed v1+v2;
    # DIAGNOSIS-ONLY after R2a v1 itself failed 0.045 at 558fcc1, codex
    # msg 1779479973262-6d7445d2 reframed variable-B as the blocker not
    # operator mixing): addition-only teens variable-B:
    #   A in [10, 19], B in [2, 9], template {A_plus_B}.
    # Phenomena: plus_no_carry + plus_carry only.
    # 75/25 stratified split: pool 80 -> 60 train + 20 held_out (3000 new /
    # 60 = 50x multiplicity, comparable to R1b1's 54x; preserves 20-row
    # unique audit). Output [12, 28]; no 3-digit class.
    # Subtraction-only R2b will follow if R2a PASSES.
    "R2a": {
        "train":     {"A_range": (10, 19), "B_range": (2, 9), "partition": "enumerate_stratified_phenom_plus"},
        "held_out":  {"A_range": (10, 19), "B_range": (2, 9), "partition": "enumerate_stratified_phenom_plus"},
    },
    # R2 (codex msg 1779476750248-2dca0aa7 after R1b2 v2 replay50 PASS at
    # c2686cc): smallest TRUE multi-digit ± bridge. Teens variable-B:
    #   A in [10, 19], B in [2, 9], templates {A_plus_B, A_minus_B}.
    #   Output stays in [1, 28] (no 3-digit class).
    # Stratify by PHENOMENON (plus_no_carry, plus_carry, minus_no_borrow,
    # minus_borrow), not just A bucket — guarantees both train and held_out
    # contain carry AND borrow cases.
    #
    # DIAGNOSIS-ONLY after R2 v1 (0.085) AND v2 n_train=8000 (0.185)
    # both failed (c2f4f8d). R2a (addition-only) is the operator-split
    # successor in the active chain. R2 stays in RUNG_NAMES at higher
    # index for backward-compat; reachable via explicit rungs= arg.
    # R2's A_plus_B rows OVERLAP R2a's by construction (same template,
    # same A/B ranges). R2 stays in DIAGNOSIS_ONLY_RUNGS until a future
    # full-R2 target passes.
    "R2": {
        "train":     {"A_range": (10, 19), "B_range": (2, 9), "partition": "enumerate_stratified_phenom"},
        "held_out":  {"A_range": (10, 19), "B_range": (2, 9), "partition": "enumerate_stratified_phenom"},
    },
    # R3 uses enumerate-partition over [0,9]² × {times}. 17×23 specifically
    # force-injected into held_out (out of [0,9]² range).
    "R3": {
        "train":     {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
        "held_out":  {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
    },
    "R4": {
        "train":     {"A_range": (0, 9), "B_range": (0, 9), "C_range": (0, 9)},
        "held_out":  {"A_range": (10, 30), "B_range": (10, 30), "C_range": (10, 30)},
    },
    # R5/R6 use template banks rather than operand ranges; R7 = GSM8k (out of scope)
}


def _enumerate_partition_r0(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R0's [0,99] operand space.

    Per codex msg 1779464341737-43a42cae: split bucket [0,9] (one-digit)
    AND bucket [10,99] (two-digit) separately with bucket-distinct
    `_stable_seed("R0_partition", seed, bucket)` so a flat shuffle can't
    accidentally segregate all 1-digit Ns onto one side.

    Result: train ~= 8 one-digit + 72 two-digit; held_out ~= 2 one-digit
    + 18 two-digit. Both splits contain both digit-length classes -> tests
    in-distribution digit-copy memorization, not OOD length generalization.

    Cross-rung-train doesn't see R0's held_out because R0's template
    `what is N?` is UNIQUE to R0 (no other rung uses single-operand
    completion).
    """
    train_set: set = set()
    held_out_set: set = set()
    # Bucket 1: one-digit [0,9]  (10 Ns -> 8 train / 2 held_out at 0.8)
    # Bucket 2: two-digit [10,99] (90 Ns -> 72 train / 18 held_out at 0.8)
    for bucket_label, lo, hi in (("one_digit", 0, 9), ("two_digit", 10, 99)):
        bucket_ns = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R0_partition", seed, bucket_label))
        rng.shuffle(bucket_ns)
        split = int(len(bucket_ns) * train_frac)
        train_set.update(bucket_ns[:split])
        held_out_set.update(bucket_ns[split:])
    return train_set, held_out_set


R1_IDENTITY_TEMPLATES = ("A_plus_0", "0_plus_A", "A_minus_0")


def _enumerate_partition_r1(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1's identity-bridge
    operand space (codex msg 1779466025267 redefinition).

    R1 v1 (single-digit ± on [0,9]^2) failed acquisition at 0.230: the
    rung combined two-operand parsing + operator binding + arithmetic
    + unseen-pair generalization into a single jump from R0. R1
    redefined as IDENTITY BRIDGE: 3 templates (A_plus_0 / 0_plus_A /
    A_minus_0), all evaluating to A. Teaches two-operand parsing +
    operator binding without yet requiring actual arithmetic
    computation; preserves R0's digit-copy primitive on the output.

    Stratification (per codex Step 1 spec): each
    (template, digit-bucket) pair partitioned 80/20 separately with a
    bucket-distinct `_stable_seed("R1_identity_partition", seed,
    template, bucket)`. Result:
      one-digit bucket A in [0,9]: 8 train + 2 held_out PER template
      two-digit bucket A in [10,99]: 72 train + 18 held_out PER template
      total: 80 + 79 + 80 = 239 train pairs
             (one fewer than 3 * (8 + 72) = 240 because "0_plus_A" drops
             A=0 to prevent row collision with "A_plus_0" + A=0; see
             row-collision-fix comment in this function's body)
             20 + 20 + 20 = 60 held_out pairs

    Both splits contain ALL 3 templates AND BOTH digit-length buckets.

    Cross-rung-train invariant: R1 identity rows have B=0 or A=0 in the
    question text; R2 train (`A in [10,99] B in [10,99]`) cannot emit
    B=0 rows; R3 uses "times" not "plus"/"minus"; R5/R6 use distinct
    surface forms. assert_no_train_holdout_overlap verifies by full row
    equality.
    """
    train_set: set = set()
    held_out_set: set = set()
    for template in R1_IDENTITY_TEMPLATES:
        for bucket_label, lo, hi in (("one_digit", 0, 9), ("two_digit", 10, 99)):
            bucket_as = list(range(lo, hi + 1))
            # ROW-COLLISION FIX: ('0_plus_A', 0) emits the same row as
            # ('A_plus_0', 0): both yield "what is 0 plus 0?" -> 0.
            # Drop A=0 from "0_plus_A" so the row exists in exactly one
            # partition cell. Multi-seed sweep confirmed 50% of random
            # seeds otherwise cross-side-collide on this row. ("A_minus_0"
            # with A=0 emits "what is 0 minus 0?" which is unique;
            # safe to keep.)
            if template == "0_plus_A" and bucket_label == "one_digit":
                bucket_as = [a for a in bucket_as if a != 0]
            rng = random.Random(
                _stable_seed("R1_identity_partition", seed, template, bucket_label)
            )
            rng.shuffle(bucket_as)
            split = int(len(bucket_as) * train_frac)
            train_set.update((template, a) for a in bucket_as[:split])
            held_out_set.update((template, a) for a in bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r1b1(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b1's single-template
    `A plus 1` operand space (codex msg 1779469364293 + 1779469638068).

    R1b1 is the falsifier-protocol split after R1b v2 failed at 0.845
    with 2x training steps ("more-updates" hypothesis falsified at
    0d152dd). Single template `what is A plus 1?` -> A+1 isolates the
    simplest sub-skill (commutative-add by 1 in `A op K` position) so
    the model can acquire it cleanly before symmetric (`1 plus A`) and
    subtraction (`A minus 1`) variations land in later sub-rungs.

    Pool: A in [1, 98]
      - drop A=0: would emit "what is 0 plus 1?" -> 1, duplicating R1
                  0_plus_A row with A=1 (cross-rung collision).
      - cap A=98: keeps output <= 99 (no new digit-length class).

    Stratification (per codex Step 1 spec): each digit-bucket
    partitioned 80/20 separately with bucket-distinct
    `_stable_seed("R1b1_partition", seed, bucket_label)` so the digit-
    length mix is preserved on both sides:
      one-digit bucket A in [1, 9]:  9 vals  ->  7 train + 2 held_out
      two-digit bucket A in [10, 98]: 89 vals -> 71 train + 18 held_out
      TOTAL: 78 train + 20 held_out integers A
            (each integer represents exactly one row since template is fixed)

    Cross-rung-train invariant: R1b1 always emits "what is A plus 1?";
    R1 emits "what is A plus 0?" / "what is 0 plus A?" / "what is A
    minus 0?" (B=0 on the additive side OR template-distinct minus);
    R2 train requires B in [10,99] so no R2 row has B=1; R3 uses
    "times" not "plus". R1b is OUT of the active chain by design
    (overlaps by construction; see _RUNG_SPEC R1b comment).
    """
    train_set: set = set()
    held_out_set: set = set()
    for bucket_label, lo, hi in (("one_digit", 1, 9), ("two_digit", 10, 98)):
        bucket_as = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R1b1_partition", seed, bucket_label))
        rng.shuffle(bucket_as)
        split = int(len(bucket_as) * train_frac)
        train_set.update(bucket_as[:split])
        held_out_set.update(bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r1b2a(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b2a's low-A
    subtraction operand space (codex msg 1779472124507 + 1779472300306
    after R1b2 FAILED at 6fd2fec).

    R1b2a is the falsifier-protocol narrower-split after R1b2 (full
    [1,99]) failed simultaneously on G1 (0.860) AND G2 R1b1 retention
    (0.880, -0.050 decay). Splits subtraction into low-A (this rung)
    + high-A (future R1b2b) to isolate operator difficulty from two-
    digit generalization. Paired with replay_ratio=0.50 to anchor R1b1
    stronger.

    Pool: A in [1, 19]
      - drop A=0: output -1 mismatches non-negative schema.
      - cap A=19: output 18 stays in [0,18]; no two-digit borrow.

    Stratification: bucket-stratified per established R1b1/R1b2 pattern:
      one-digit bucket A in [1, 9]:   9 vals  -> 7 train + 2 held_out
      teen bucket     A in [10, 19]: 10 vals -> 8 train + 2 held_out
      TOTAL: 15 train + 4 held_out integers A

    Codex guardrail msg 1779472239175: 2-row held_out at [1,9] alone
    was too weak; [1,19] gives 4 unique held_out (one_digit + teen)
    enabling unique-heldout G1 audit (must be 4/4) alongside the
    standard oversampled exact rate.

    Cross-rung-train invariant: R1b2a emits `what is A minus 1?` over
    A in [1,19]. R1 emits `A minus 0` (B=0; disjoint by B-value); R1b1
    emits `A plus 1` (disjoint by operator). R2 train requires
    B in [10,99] so no R2 row has B=1. R1b2 (failed) and R1b
    (diagnosis-only) both emit `A_minus_1` over A in [1,99] which
    OVERLAPS R1b2a's [1,19] strict subset by construction; R1b2 and
    R1b both stay excluded from build_rung_splits default.
    """
    train_set: set = set()
    held_out_set: set = set()
    for bucket_label, lo, hi in (("one_digit", 1, 9), ("teen", 10, 19)):
        bucket_as = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R1b2a_partition", seed, bucket_label))
        rng.shuffle(bucket_as)
        split = int(len(bucket_as) * train_frac)
        train_set.update(bucket_as[:split])
        held_out_set.update(bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r1b2(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b2's single-template
    `A minus 1` operand space (codex msg 1779471073874 + 1779471212090).

    R1b2 isolates the subtraction operator at the same operand position
    as R1b1 (`A op K` shape, K=1). Falsifier: does `-` acquire comparably
    to `+1` once position symmetry is held fixed?

    Pool: A in [1, 99]
      - drop A=0: output would be -1 (negative; schema mismatches non-
                  negative integer answers).
      - A=99 allowed: output 98 stays in [0,99] (no digit-length class
                      change — unlike R1b1 where A=99 would push output
                      to 100; subtraction can't overflow upward).

    Stratification: each digit-bucket partitioned 80/20 separately with
    bucket-distinct `_stable_seed("R1b2_partition", seed, bucket_label)`:
      one-digit bucket A in [1, 9]:   9 vals  -> 7 train + 2 held_out
      two-digit bucket A in [10, 99]: 90 vals -> 72 train + 18 held_out
      TOTAL: 79 train + 20 held_out integers A

    Cross-rung-train invariant: R1b2 always emits `what is A minus 1?`;
    R1 emits `A minus 0` (B=0); R1b1 emits `A plus 1` (operator word
    distinct); R2 train requires B in [10,99] so no R2 row has B=1.
    R1b's diagnosis-only A_minus_1 template OVERLAPS R1b2 by
    construction (same template, same A range) — R1b stays excluded
    from build_rung_splits default per established active-chain policy.
    """
    train_set: set = set()
    held_out_set: set = set()
    for bucket_label, lo, hi in (("one_digit", 1, 9), ("two_digit", 10, 99)):
        bucket_as = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R1b2_partition", seed, bucket_label))
        rng.shuffle(bucket_as)
        split = int(len(bucket_as) * train_frac)
        train_set.update(bucket_as[:split])
        held_out_set.update(bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r1b3(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b3's constant-K=2
    operand space (codex msg 1779479973262-6d7445d2 after R2a v1 failed
    0.045 at 558fcc1; variable-B reframed as the structural blocker).

    R1b3 extends the locked-piece pattern (constant-B single-template,
    proven via R1b1 K=1 and R1b2 K=-1) to K=2 BEFORE attempting
    variable-B again.

    Pool: A in [1, 97]
      - drop A=0: avoids collision with R1's 0_plus_A (A=2 emits
                  "what is 0 plus 2?" -> 2; R1 has B=0 not B=2, so no
                  collision by B-value; A=0 still excluded for symmetry
                  with R1b1's drop and to avoid emitting "what is 0 plus
                  2?" which the model might confuse with R1 templates).
      - cap A=97: keeps output A+2 <= 99 (no 3-digit class).

    Stratification: bucket-stratified per R1b1 pattern with
    bucket-distinct `_stable_seed("R1b3_partition", seed, bucket_label)`.
      one_digit [1, 9]:    9 vals -> 7 train + 2 held_out
      two_digit [10, 97]: 88 vals -> 70 train + 18 held_out
      TOTAL:              97 vals -> 77 train + 20 held_out integers A

    Cross-rung-train invariant: R1b3 emits `what is A plus 2?` with
    B=2. R0 has no second operand. R1 has B=0 (additive side). R1b1
    has B=1 (plus). R1b2 has B=1 (minus). All disjoint by B-value.
    R2a (diagnosis-only) has variable B in [2,9] over A in [10,19];
    overlaps R1b3 only for B=2, A in [10,19] -- R2a stays excluded
    from active chain via DIAGNOSIS_ONLY_RUNGS.
    """
    train_set: set = set()
    held_out_set: set = set()
    for bucket_label, lo, hi in (("one_digit", 1, 9), ("two_digit", 10, 97)):
        bucket_as = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R1b3_partition", seed, bucket_label))
        rng.shuffle(bucket_as)
        split = int(len(bucket_as) * train_frac)
        train_set.update(bucket_as[:split])
        held_out_set.update(bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r1b4(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b4's constant-K=3
    addition operand space (codex msg 1779482125661-b2c0ca2a after R1b3
    v2 schedule PASS at 175d327).

    Continues locked constant-B single-template pattern from K=1 (R1b1)
    -> K=-1 (R1b2) -> K=2 (R1b3) -> K=3 (this rung).

    Pool: A in [1, 96]
      - drop A=0: symmetric with R1b1/R1b3 drops.
      - cap A=96: keeps output A+3 <= 99 (no 3-digit class).

    Stratification: bucket-stratified per R1b1/R1b3 pattern with
    bucket-distinct `_stable_seed("R1b4_partition", seed, bucket_label)`.
      one_digit [1, 9]:    9 vals -> 7 train + 2 held_out
      two_digit [10, 96]: 87 vals -> 69 train + 18 held_out
      TOTAL:              96 vals -> 76 train + 20 held_out integers A

    Cross-rung-train invariant: B=3 disjoint from priors R0 (no B), R1
    (B=0), R1b1 (B=1), R1b2 (B=1 minus), R1b3 (B=2). No collision with
    diagnosis-only R2a [10,19]×[2,9] except for B=3 A in [10,19] subset;
    R2a stays out of active chain via DIAGNOSIS_ONLY_RUNGS.
    """
    train_set: set = set()
    held_out_set: set = set()
    for bucket_label, lo, hi in (("one_digit", 1, 9), ("two_digit", 10, 96)):
        bucket_as = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R1b4_partition", seed, bucket_label))
        rng.shuffle(bucket_as)
        split = int(len(bucket_as) * train_frac)
        train_set.update(bucket_as[:split])
        held_out_set.update(bucket_as[split:])
    return train_set, held_out_set


R1B_TEMPLATES = ("A_plus_1", "1_plus_A", "A_minus_1")


def _enumerate_partition_r1b(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b's ±1 operand space
    (codex msg 1779467425298 after R1 identity-bridge pass).

    R1b is the minimal-arithmetic bridge from R1 identity to actual
    single-digit ±1. Three templates with per-template A constraints
    chosen to prevent cross-template AND cross-rung row collisions:

      A_plus_1:  A in [1, 98]    -- "what is A plus 1?"  -> A+1
                                 -- drop A=0: would emit "what is 0 plus 1?" -> 1,
                                    duplicating R1 0_plus_A with A=1
                                 -- cap A=98: keeps output <= 99 (no 3-digit class)
      1_plus_A:  A in [2, 98]    -- "what is 1 plus A?"  -> A+1
                                 -- drop A=0: would emit "what is 1 plus 0?" -> 1,
                                    duplicating R1 A_plus_0 with A=1
                                 -- drop A=1: would emit "what is 1 plus 1?" -> 2,
                                    duplicating intra-R1b A_plus_1 with A=1
                                 -- cap A=98: keeps output <= 99
      A_minus_1: A in [1, 99]    -- "what is A minus 1?" -> A-1
                                 -- drop A=0: would emit "what is 0 minus 1?" -> -1
                                    (negative output, schema mismatch)

    Pool sizes per template:
      A_plus_1:  [1,9]   ( 9 vals) -> 7 train + 2 held;
                 [10,98] (89 vals) -> 71 train + 18 held; total 78 train + 20 held
      1_plus_A:  [2,9]   ( 8 vals) -> 6 train + 2 held;
                 [10,98] (89 vals) -> 71 train + 18 held; total 77 train + 20 held
      A_minus_1: [1,9]   ( 9 vals) -> 7 train + 2 held;
                 [10,99] (90 vals) -> 72 train + 18 held; total 79 train + 20 held
      TOTAL: 234 train + 60 held_out pairs

    All splits contain ALL 3 templates AND BOTH digit-length buckets.
    """
    train_set: set = set()
    held_out_set: set = set()

    template_specs = {
        "A_plus_1":  {"one_digit": list(range(1, 10)),  "two_digit": list(range(10, 99))},
        "1_plus_A":  {"one_digit": list(range(2, 10)),  "two_digit": list(range(10, 99))},
        "A_minus_1": {"one_digit": list(range(1, 10)),  "two_digit": list(range(10, 100))},
    }

    for template in R1B_TEMPLATES:
        for bucket_label in ("one_digit", "two_digit"):
            bucket_as = list(template_specs[template][bucket_label])
            rng = random.Random(
                _stable_seed("R1b_partition", seed, template, bucket_label)
            )
            rng.shuffle(bucket_as)
            split = int(len(bucket_as) * train_frac)
            train_set.update((template, a) for a in bucket_as[:split])
            held_out_set.update((template, a) for a in bucket_as[split:])
    return train_set, held_out_set


def _enumerate_partition_r3(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Deterministic 80/20 partition of R3's operand space [0,9]² × {times}.
    17×23 canonical force-injected into held_out (out-of-range A=17 B=23)."""
    pairs = [(A, B) for A in range(10) for B in range(10)]
    rng = random.Random(_stable_seed("R3_partition", seed))
    rng.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train = set(pairs[:split])
    held_out = set(pairs[split:])
    # Force 17×23 into held_out (canonical mastery gate)
    held_out.add((17, 23))
    return train, held_out


# Phrasings for R5 paraphrase rung
_R5_PHRASINGS_TRAIN = [
    "what is {expr}?",
    "compute {expr}",
    "{expr} equals what?",
]
_R5_PHRASINGS_HELDOUT = [
    "what's {expr}",
    "calculate {expr}",
    "the value of {expr} is what?",
]

# Templates for R6 word problems
_R6_TEMPLATES_TRAIN = [
    ("Alice has {A} apples. Bob gives her {B} more. How many apples does Alice have?",  # +
     lambda A, B: A + B),
    ("There are {A} cookies in a jar. {B} are eaten. How many cookies are left?",        # -
     lambda A, B: A - B),
    ("A box contains {A} items. There are {B} such boxes. How many items in total?",     # *
     lambda A, B: A * B),
]
_R6_TEMPLATES_HELDOUT = [
    ("Carlos collected {A} stamps. He buys {B} more at a fair. How many stamps now?",    # +
     lambda A, B: A + B),
    ("A library has {A} books. {B} are checked out. How many remain?",                   # -
     lambda A, B: A - B),
    ("Each shelf holds {A} books. There are {B} shelves. How many books in total?",      # *
     lambda A, B: A * B),
]


def _make_rng(rung: str, seed: int, split: str) -> random.Random:
    """Stable per-(rung, seed, split) RNG. Different splits get distinct
    seeds derived from the input seed so train and held_out never share
    state."""
    salt = {"train": 0, "held_out": 17}[split]
    return random.Random(_stable_seed(rung, seed, salt))


def _gen_r0(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R0: digit copy `what is N? -> N`.

    Stratified in-distribution partition over [0,99] (codex msg
    1779464341737-43a42cae). Train pool ~80 Ns (8 one-digit + 72 two-digit);
    held_out pool ~20 Ns (2 one-digit + 18 two-digit). Both contain both
    digit-length classes. NO operand >= 100 in either split."""
    train_pool, held_out_pool = _enumerate_partition_r0(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order before rng.choice
    out = []
    while len(out) < n:
        N = rng.choice(pool_list)
        out.append({"question": f"what is {N}?", "expected": N, "rung": "R0"})
    return out


def _gen_r1(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1 identity-bridge (codex msg 1779466025267 redefinition).

    3 templates × A in [0,99] stratified by (template, digit-bucket):
      `what is A plus 0?` -> A     (template A_plus_0)
      `what is 0 plus A?` -> A     (template 0_plus_A)
      `what is A minus 0?` -> A    (template A_minus_0)

    Output = A in every case (preserves R0 digit-copy primitive on the
    answer side). Trains two-operand parsing + operator binding without
    yet requiring arithmetic computation.

    Train pool = 239 (template, A) pairs; held_out pool = 60. The
    "0_plus_A" template drops A=0 to prevent row collision with
    "A_plus_0" + A=0 (both emit "what is 0 plus 0?"). Both splits
    contain ALL 3 templates AND BOTH digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order before rng.choice
    out = []
    while len(out) < n:
        template, A = rng.choice(pool_list)
        if template == "A_plus_0":
            q = f"what is {A} plus 0?"
        elif template == "0_plus_A":
            q = f"what is 0 plus {A}?"
        elif template == "A_minus_0":
            q = f"what is {A} minus 0?"
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown R1 identity template: {template!r}")
        out.append({"question": q, "expected": A, "rung": "R1"})
    return out


def _gen_r1b1(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b1 single-template `A plus 1` (codex msg 1779469364293
    falsifier-protocol split after R1b v2 failed at 0.845).

    Single template `what is A plus 1?` -> A+1. A in [1, 98]
    bucket-stratified. Output ∈ [2, 99]. Designed disjoint from R1
    (R1 has B=0 on additive side; R1b1 has B=1) and disjoint from
    R2+ (R2 train requires B>=10).

    Train pool = 78 integers A; held_out pool = 20 integers A. Both
    splits contain BOTH digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1b1(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 1?"
        expected = A + 1
        out.append({"question": q, "expected": expected, "rung": "R1b1"})
    return out


def _gen_r1b2a(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b2a low-A subtraction (codex msg 1779472124507 + 1779472300306
    after R1b2 FAIL at 6fd2fec).

    Single template `what is A minus 1?` -> A-1. A in [1, 19]
    bucket-stratified (one_digit + teen). Output ∈ [0, 18]. Designed
    disjoint from R1 (R1 B=0), R1b1 (plus vs minus), R2+ (B≥10);
    overlaps R1b2/R1b A_minus_1 by construction (both diagnosis-only).

    Train pool = 15 integers A; held_out pool = 4 integers A. Both
    splits contain BOTH digit-length buckets (one_digit + teen)."""
    train_pool, held_out_pool = _enumerate_partition_r1b2a(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} minus 1?"
        expected = A - 1
        out.append({"question": q, "expected": expected, "rung": "R1b2a"})
    return out


def _gen_r1b2(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b2 single-template `A minus 1` (codex msg 1779471073874 +
    1779471212090 after R1b1 PASS at 66b9747).

    Single template `what is A minus 1?` -> A-1. A in [1, 99]
    bucket-stratified. Output ∈ [0, 98]. Designed disjoint from R1
    (R1 has B=0 on additive side; R1b2 has B=1 on subtractive side),
    R1b1 (R1b1 uses "plus"; R1b2 uses "minus"), and R2+ (R2 train
    requires B>=10).

    Train pool = 79 integers A; held_out pool = 20 integers A. Both
    splits contain BOTH digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1b2(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} minus 1?"
        expected = A - 1
        out.append({"question": q, "expected": expected, "rung": "R1b2"})
    return out


def _gen_r1b4(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b4 constant K=3 addition (codex msg 1779482125661-b2c0ca2a after
    R1b3 v2 schedule PASS at 175d327).

    Single template `what is A plus 3?` -> A+3. A in [1, 96]
    bucket-stratified. Output [4, 99]; no 3-digit class. Continues
    locked constant-B pattern (K=1, K=-1, K=2 all passed) to K=3.

    Train pool = 76 integers A; held_out pool = 20."""
    train_pool, held_out_pool = _enumerate_partition_r1b4(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 3?"
        expected = A + 3
        out.append({"question": q, "expected": expected, "rung": "R1b4"})
    return out


def _enumerate_partition_r1b4v2(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic partition for R1b4v2's K=3 addition with
    one-digit-EXHAUSTIVE train support (codex msg 1779483673737-20ff22ab
    after R1b4 v1 fail at 7b53368).

    Provenance-preserving fix to R1b4 v1's measurement bug: R1b4 v1
    split one_digit 7/2 train/held, then `probe_curriculum` sampled
    eval_cap=200 via `rng.choice(pool_list)`, so the 2 one_digit
    heldout rows were each measured ~22× — repeated-sample weighting
    of a tiny bucket, not generalization signal. Result: standard
    metric 0.885 < G1 0.90 by 0.015 while two_digit hit 18/18 perfect.

    R1b4v2 partition:
      one_digit [1, 9]:    9 vals -> 9 EXHAUSTIVE train + 0 held_out
      two_digit [10, 96]: 87 vals -> 69 train + 18 held_out (80/20)
      TOTAL:              96 vals -> 78 train + 18 held_out

    Held_out contains ZERO one_digit rows by design — one_digit mastery
    gated via separate deterministic 9-row exhaustive audit (see
    `r1b4v2_one_digit_audit_rows`). two_digit retains the standard
    80/20 stratified split with bucket-distinct seed
    `_stable_seed("R1b4v2_partition", seed, "two_digit")`.
    """
    train_set: set = set()
    held_out_set: set = set()

    # one_digit: exhaustive into train, ZERO held_out
    for A in range(1, 10):
        train_set.add(A)

    # two_digit: standard 80/20 stratified split
    bucket_as = list(range(10, 97))
    rng = random.Random(_stable_seed("R1b4v2_partition", seed, "two_digit"))
    rng.shuffle(bucket_as)
    split = int(len(bucket_as) * train_frac)
    train_set.update(bucket_as[:split])
    held_out_set.update(bucket_as[split:])

    return train_set, held_out_set


def r1b4v2_one_digit_audit_rows(seed: int = 42) -> list[dict]:
    """Deterministic 9-row exhaustive audit of R1b4v2 one_digit support
    (A in [1, 9], `what is A plus 3?` -> A+3).

    Served separately from heldout (which is two_digit-only by R1b4v2
    design). Mastery on this audit means 9/9 — finite-domain exhaustive
    check, not repeated-sample probe. Codex msg 1779483673737-20ff22ab
    spec: "since the finite domain has 9 cases, mastery means 9/9
    rather than pretending two unseen rows are a robust generalization
    estimate."

    The `seed` arg is accepted for API symmetry with the generator
    functions but the row contents are seed-invariant (the rows ARE
    the full domain, no sampling). It's available so callers can pass
    the trainer seed for log/audit traceability without branching.

    Returns list of {"question", "expected", "rung"} dicts sorted by A.
    """
    del seed  # accepted for API symmetry; row contents are exhaustive
    return [
        {"question": f"what is {A} plus 3?", "expected": A + 3, "rung": "R1b4v2"}
        for A in range(1, 10)
    ]


def _gen_r1b4v2(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b4v2 constant K=3 addition, one-digit-EXHAUSTIVE partition (codex
    msg 1779483673737-20ff22ab after R1b4 v1 fail at 7b53368).

    Same question/output as R1b4 (`what is A plus 3?` -> A+3) but
    measurement/support redesign: one_digit A=1..9 ALL in train (9
    rows); two_digit A=10..96 80/20 stratified. Heldout is two_digit
    ONLY (18 rows). one_digit mastery gated separately via
    `r1b4v2_one_digit_audit_rows`.

    Train pool = 78 integers A (9 one_digit + 69 two_digit);
    held_out pool = 18 (all two_digit).
    """
    train_pool, held_out_pool = _enumerate_partition_r1b4v2(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 3?"
        expected = A + 3
        out.append({"question": q, "expected": expected, "rung": "R1b4v2"})
    return out


def _enumerate_partition_r1b5(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic partition for R1b5's K=4 addition (codex
    msgs 1779488238721-49f03cc9 + 1779523412979-ff88b885 after R1b4v2
    advance at b368b81).

    Continues locked constant-K single-template jigsaw to K=4. Bakes in
    R1b4v2 measurement/support lessons FROM START:
      one_digit A=1..9 EXHAUSTIVE train (NO thin-pool heldout)
      two_digit A=10..95 CARRY-STRATIFIED 80/20 (NEW dimension)

    Carry stratification (units+4 >= 10, units in {6,7,8,9}):
      carry-bucket A:        32 vals -> 25 train + 7 held_out
      non-carry-bucket A:    54 vals -> 43 train + 11 held_out
      TOTAL two_digit:       86 vals -> 68 train + 18 held_out

    Guaranteed heldout shape: 7 carry + 11 non-carry. Designed to make
    the strict-gate carry signal sample-stable across seeds (vs
    R1b4v2's accidental composition that varied per seed).

    Bucket-distinct seeds:
      _stable_seed("R1b5_partition", seed, "carry")
      _stable_seed("R1b5_partition", seed, "non_carry")
    """
    train_set: set = set()
    held_out_set: set = set()

    # one_digit: exhaustive into train, ZERO held_out (R1b4v2 lesson)
    for A in range(1, 10):
        train_set.add(A)

    # two_digit: CARRY-stratified split
    carry_as = [A for A in range(10, 96) if (A % 10) in (6, 7, 8, 9)]
    non_carry_as = [A for A in range(10, 96) if (A % 10) < 6]
    # Sanity: 32 + 54 = 86 exhaustively covers [10, 95]
    assert len(carry_as) == 32, f"R1b5 carry bucket size: {len(carry_as)}"
    assert len(non_carry_as) == 54, f"R1b5 non-carry bucket size: {len(non_carry_as)}"

    rng = random.Random(_stable_seed("R1b5_partition", seed, "carry"))
    rng.shuffle(carry_as)
    split = int(len(carry_as) * train_frac)  # 32 * 0.8 = 25
    train_set.update(carry_as[:split])
    held_out_set.update(carry_as[split:])

    rng = random.Random(_stable_seed("R1b5_partition", seed, "non_carry"))
    rng.shuffle(non_carry_as)
    split = int(len(non_carry_as) * train_frac)  # 54 * 0.8 = 43
    train_set.update(non_carry_as[:split])
    held_out_set.update(non_carry_as[split:])

    return train_set, held_out_set


def r1b5_one_digit_audit_rows(seed: int = 42) -> list[dict]:
    """Deterministic 9-row exhaustive audit of R1b5 one_digit support
    (A in [1, 9], `what is A plus 4?` -> A+4).

    Codex msg 1779523412979-ff88b885 spec: same shape as
    `r1b4v2_one_digit_audit_rows` but K=4 instead of K=3. Served
    separately from heldout (which is two_digit-only by R1b5 design).
    Mastery = 9/9 finite-domain exhaustive check.

    Seed-invariant by construction (rows ARE the full domain). Seed
    arg accepted for API symmetry with the generator functions.
    """
    del seed  # accepted for API symmetry; row contents are exhaustive
    return [
        {"question": f"what is {A} plus 4?", "expected": A + 4, "rung": "R1b5"}
        for A in range(1, 10)
    ]


def _gen_r1b5(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b5 constant K=4 addition, carry-stratified partition (codex msgs
    1779488238721-49f03cc9 + 1779523412979-ff88b885 after R1b4v2 advance
    at b368b81).

    Single template `what is A plus 4?` -> A+4. one_digit A=1..9
    exhaustive in train (per R1b4v2 lesson); two_digit A=10..95
    carry-stratified 80/20 (NEW dimension beyond R1b4v2).

    Train pool = 77 integers A (9 one_digit + 25 carry + 43 non-carry);
    held_out pool = 18 (7 carry + 11 non-carry, all two_digit).
    """
    train_pool, held_out_pool = _enumerate_partition_r1b5(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 4?"
        expected = A + 4
        out.append({"question": q, "expected": expected, "rung": "R1b5"})
    return out


def _enumerate_partition_r1b6(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic partition for R1b6's K=5 addition (codex
    msg 1779545956176-4a8cfc3e +1 K=5 naming after gabe greenlight relay
    1779545575582-7c52a912 of verbatim "ok implement, full prov").

    Mirrors R1b5 design at K=5. Bakes in R1b4v2/R1b5 lessons FROM START:
      one_digit A=1..9 EXHAUSTIVE train (NO thin-pool heldout)
      two_digit A=10..94 CARRY-STRATIFIED 80/20

    Carry stratification (units+5 >= 10, units in {5,6,7,8,9}):
      carry-bucket A:        40 vals -> 32 train + 8 held_out
      non-carry-bucket A:    45 vals -> 36 train + 9 held_out
      TOTAL two_digit:       85 vals -> 68 train + 17 held_out
    Total (with one_digit):  94 vals -> 77 train + 17 held_out

    Guaranteed heldout shape: 8 carry + 9 non-carry. 8 carry vs R1b5's
    7 is a side-effect of K=5 widening the carry support set (5 carry
    units vs 4 at K=4); design intent is the same shape, not the same
    count.

    Bucket-distinct seeds:
      _stable_seed("R1b6_partition", seed, "carry")
      _stable_seed("R1b6_partition", seed, "non_carry")
    """
    train_set: set = set()
    held_out_set: set = set()

    # one_digit: exhaustive into train, ZERO held_out (R1b4v2 lesson)
    for A in range(1, 10):
        train_set.add(A)

    # two_digit: CARRY-stratified split for K=5
    carry_as = [A for A in range(10, 95) if (A % 10) in (5, 6, 7, 8, 9)]
    non_carry_as = [A for A in range(10, 95) if (A % 10) < 5]
    # Sanity: 40 + 45 = 85 exhaustively covers [10, 94]
    assert len(carry_as) == 40, f"R1b6 carry bucket size: {len(carry_as)}"
    assert len(non_carry_as) == 45, f"R1b6 non-carry bucket size: {len(non_carry_as)}"

    rng = random.Random(_stable_seed("R1b6_partition", seed, "carry"))
    rng.shuffle(carry_as)
    split = int(len(carry_as) * train_frac)  # 40 * 0.8 = 32
    train_set.update(carry_as[:split])
    held_out_set.update(carry_as[split:])

    rng = random.Random(_stable_seed("R1b6_partition", seed, "non_carry"))
    rng.shuffle(non_carry_as)
    split = int(len(non_carry_as) * train_frac)  # 45 * 0.8 = 36
    train_set.update(non_carry_as[:split])
    held_out_set.update(non_carry_as[split:])

    return train_set, held_out_set


def r1b6_one_digit_audit_rows(seed: int = 42) -> list[dict]:
    """Deterministic 9-row exhaustive audit of R1b6 one_digit support
    (A in [1, 9], `what is A plus 5?` -> A+5).

    Codex msg 1779545956176-4a8cfc3e: same shape as
    `r1b5_one_digit_audit_rows` but K=5 instead of K=4. Served
    separately from heldout (which is two_digit-only by R1b6 design).
    Mastery = 9/9 finite-domain exhaustive check.

    Seed-invariant by construction (rows ARE the full domain). Seed
    arg accepted for API symmetry with the generator functions.
    """
    del seed  # accepted for API symmetry; row contents are exhaustive
    return [
        {"question": f"what is {A} plus 5?", "expected": A + 5, "rung": "R1b6"}
        for A in range(1, 10)
    ]


def _gen_r1b6(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b6 constant K=5 addition, carry-stratified partition (codex msg
    1779545956176-4a8cfc3e +1 K=5 naming after gabe greenlight relay
    1779545575582-7c52a912).

    Single template `what is A plus 5?` -> A+5. one_digit A=1..9
    exhaustive in train; two_digit A=10..94 carry-stratified 80/20
    with carry-units widened to {5,6,7,8,9} since K=5.

    Train pool = 77 integers A (9 one_digit + 32 carry + 36 non-carry);
    held_out pool = 17 (8 carry + 9 non-carry, all two_digit).
    """
    train_pool, held_out_pool = _enumerate_partition_r1b6(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 5?"
        expected = A + 5
        out.append({"question": q, "expected": expected, "rung": "R1b6"})
    return out


def _enumerate_partition_r1b7(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic partition for R1b7's K=6 addition (codex
    msg 1779547753761-5711d790 +1 K=6; rebased onto R1b2-repair commit
    9c8f800 per codex msg 1779549330637-876c3453 after R1b6 candidate
    full-chain baseline revealed R1b2=0.78 pre-existing gap; durable
    gabe provenance relay 1779547541812).

    Mirrors R1b5/R1b6 design at K=6. Bakes in R1b4v2/R1b5/R1b6 lessons:
      one_digit A=1..9 EXHAUSTIVE train (NO thin-pool heldout)
      two_digit A=10..93 CARRY-STRATIFIED 80/20

    Carry stratification (units+6 >= 10, units in {4,5,6,7,8,9}):
      carry-bucket A:        48 vals -> 38 train + 10 held_out
      non-carry-bucket A:    36 vals -> 28 train +  8 held_out
      TOTAL two_digit:       84 vals -> 66 train + 18 held_out
    Total (with one_digit):  93 vals -> 75 train + 18 held_out

    Guaranteed heldout shape: 10 carry + 8 non-carry. 10 carry vs
    R1b6's 8 is a side-effect of K=6 widening the carry support set
    (6 carry units vs 5 at K=5); design intent is the same shape,
    not the same count.

    Bucket-distinct seeds:
      _stable_seed("R1b7_partition", seed, "carry")
      _stable_seed("R1b7_partition", seed, "non_carry")
    """
    train_set: set = set()
    held_out_set: set = set()

    # one_digit: exhaustive into train, ZERO held_out (R1b4v2 lesson)
    for A in range(1, 10):
        train_set.add(A)

    # two_digit: CARRY-stratified split for K=6
    carry_as = [A for A in range(10, 94) if (A % 10) in (4, 5, 6, 7, 8, 9)]
    non_carry_as = [A for A in range(10, 94) if (A % 10) < 4]
    # Sanity: 48 + 36 = 84 exhaustively covers [10, 93]
    assert len(carry_as) == 48, f"R1b7 carry bucket size: {len(carry_as)}"
    assert len(non_carry_as) == 36, f"R1b7 non-carry bucket size: {len(non_carry_as)}"

    rng = random.Random(_stable_seed("R1b7_partition", seed, "carry"))
    rng.shuffle(carry_as)
    split = int(len(carry_as) * train_frac)  # 48 * 0.8 = 38
    train_set.update(carry_as[:split])
    held_out_set.update(carry_as[split:])

    rng = random.Random(_stable_seed("R1b7_partition", seed, "non_carry"))
    rng.shuffle(non_carry_as)
    split = int(len(non_carry_as) * train_frac)  # 36 * 0.8 = 28
    train_set.update(non_carry_as[:split])
    held_out_set.update(non_carry_as[split:])

    return train_set, held_out_set


def r1b7_one_digit_audit_rows(seed: int = 42) -> list[dict]:
    """Deterministic 9-row exhaustive audit of R1b7 one_digit support
    (A in [1, 9], `what is A plus 6?` -> A+6).

    Codex msg 1779547753761-5711d790: same shape as
    `r1b6_one_digit_audit_rows` but K=6 instead of K=5. Served
    separately from heldout (which is two_digit-only by R1b7 design).
    Mastery = 9/9 finite-domain exhaustive check.

    Seed-invariant by construction (rows ARE the full domain). Seed
    arg accepted for API symmetry with the generator functions.
    """
    del seed  # accepted for API symmetry; row contents are exhaustive
    return [
        {"question": f"what is {A} plus 6?", "expected": A + 6, "rung": "R1b7"}
        for A in range(1, 10)
    ]


def _enumerate_partition_r1b8(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic partition for R1b8's K=7 addition (codex
    msg 1779550489408-f40f66ab +1 K=7 after R1b7 commit 682659b ADVANCED
    via R1b2-retained chain + A0 exhaustive audit 1071/1072 PASS).

    Mirrors R1b5/R1b6/R1b7 design at K=7. Bakes in lessons:
      one_digit A=1..9 EXHAUSTIVE train (NO thin-pool heldout)
      two_digit A=10..92 CARRY-STRATIFIED 80/20

    Carry stratification (units+7 >= 10, units in {3,4,5,6,7,8,9}):
      carry-bucket A:        56 vals -> 44 train + 12 held_out
      non-carry-bucket A:    27 vals -> 21 train +  6 held_out
      TOTAL two_digit:       83 vals -> 65 train + 18 held_out
    Total (with one_digit):  92 vals -> 74 train + 18 held_out

    Guaranteed heldout shape: 12 carry + 6 non-carry. 12 carry vs R1b7's
    10 is a side-effect of K=7 widening the carry support set (7 carry
    units vs 6 at K=6); design intent is the same shape, not the same count.

    Bucket-distinct seeds:
      _stable_seed("R1b8_partition", seed, "carry")
      _stable_seed("R1b8_partition", seed, "non_carry")
    """
    train_set: set = set()
    held_out_set: set = set()

    for A in range(1, 10):
        train_set.add(A)

    carry_as = [A for A in range(10, 93) if (A % 10) in (3, 4, 5, 6, 7, 8, 9)]
    non_carry_as = [A for A in range(10, 93) if (A % 10) < 3]
    assert len(carry_as) == 56, f"R1b8 carry bucket size: {len(carry_as)}"
    assert len(non_carry_as) == 27, f"R1b8 non-carry bucket size: {len(non_carry_as)}"

    rng = random.Random(_stable_seed("R1b8_partition", seed, "carry"))
    rng.shuffle(carry_as)
    split = int(len(carry_as) * train_frac)  # 56 * 0.8 = 44
    train_set.update(carry_as[:split])
    held_out_set.update(carry_as[split:])

    rng = random.Random(_stable_seed("R1b8_partition", seed, "non_carry"))
    rng.shuffle(non_carry_as)
    split = int(len(non_carry_as) * train_frac)  # 27 * 0.8 = 21
    train_set.update(non_carry_as[:split])
    held_out_set.update(non_carry_as[split:])

    return train_set, held_out_set


def r1b8_one_digit_audit_rows(seed: int = 42) -> list[dict]:
    """Deterministic 9-row exhaustive audit of R1b8 one_digit support
    (A in [1, 9], `what is A plus 7?` -> A+7).

    Codex msg 1779550489408-f40f66ab: same shape as r1b7_one_digit_audit_rows
    but K=7 instead of K=6. Mastery = 9/9 finite-domain exhaustive check.
    Seed-invariant by construction.
    """
    del seed
    return [
        {"question": f"what is {A} plus 7?", "expected": A + 7, "rung": "R1b8"}
        for A in range(1, 10)
    ]


def _gen_r1b8(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b8 constant K=7 addition, carry-stratified partition (codex msg
    1779550489408-f40f66ab +1 K=7 after R1b7 commit 682659b).

    Single template `what is A plus 7?` -> A+7. one_digit A=1..9
    exhaustive in train; two_digit A=10..92 carry-stratified 80/20
    with carry-units widened to {3,4,5,6,7,8,9} since K=7.

    Train pool = 74 integers A (9 one_digit + 44 carry + 21 non-carry);
    held_out pool = 18 (12 carry + 6 non-carry, all two_digit).
    """
    train_pool, held_out_pool = _enumerate_partition_r1b8(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 7?"
        expected = A + 7
        out.append({"question": q, "expected": expected, "rung": "R1b8"})
    return out


def _gen_r1b7(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b7 constant K=6 addition, carry-stratified partition (codex msg
    1779547753761-5711d790 +1 K=6; rebased onto R1b2-repair commit
    9c8f800 per codex msg 1779549330637-876c3453 with R1b2 explicit
    hard retention gate).

    Single template `what is A plus 6?` -> A+6. one_digit A=1..9
    exhaustive in train; two_digit A=10..93 carry-stratified 80/20
    with carry-units widened to {4,5,6,7,8,9} since K=6.

    Train pool = 75 integers A (9 one_digit + 38 carry + 28 non-carry);
    held_out pool = 18 (10 carry + 8 non-carry, all two_digit).
    """
    train_pool, held_out_pool = _enumerate_partition_r1b7(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 6?"
        expected = A + 6
        out.append({"question": q, "expected": expected, "rung": "R1b7"})
    return out


def _gen_r1b(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b ±1 (codex msg 1779467425298 after R1 identity pass at c6e94578).

    3 templates × stratified-partitioned A range:
      `what is A plus 1?`  -> A+1  (A in [1,98])
      `what is 1 plus A?`  -> A+1  (A in [2,98])
      `what is A minus 1?` -> A-1  (A in [1,99])

    Output stays in [0,99]; no new digit-length class. Train pool = 234
    (template, A) pairs; held_out pool = 60. Both splits contain all 3
    templates + both digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1b(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        template, A = rng.choice(pool_list)
        if template == "A_plus_1":
            q = f"what is {A} plus 1?"
            expected = A + 1
        elif template == "1_plus_A":
            q = f"what is 1 plus {A}?"
            expected = A + 1
        elif template == "A_minus_1":
            q = f"what is {A} minus 1?"
            expected = A - 1
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown R1b template: {template!r}")
        out.append({"question": q, "expected": expected, "rung": "R1b"})
    return out


R2A_PHENOMENA = ("plus_no_carry", "plus_carry")


def _r2a_phenomenon(A: int, B: int) -> str:
    """Classify (A, B) as plus_no_carry or plus_carry for R2a's
    addition-only stratification (codex msg 1779478819906-0e30503e)."""
    return "plus_carry" if (A % 10) + B >= 10 else "plus_no_carry"


def _enumerate_partition_r2a(seed: int, train_frac: float = 0.75) -> tuple[set, set]:
    """Stratified deterministic 75/25 partition of R2a's teens addition-only
    operand space (codex msg 1779478819906-0e30503e after full R2 failed v1+v2).

    R2a is the operator-split successor to failed full teens ± R2.
    Addition only: template `what is A plus B?`, A in [10,19], B in [2,9].
    Phenomenon-stratified across (plus_no_carry, plus_carry); 75/25 split
    chosen (vs prior rungs' 80/20) to leave 20 unique held_out rows while
    keeping multiplicity comparable to R1b1's 54x.

    Pool sizes per phenomenon:
      plus_no_carry: 36 -> 27 train + 9 held_out  (36 * 0.75 = 27.0)
      plus_carry:    44 -> 33 train + 11 held_out (44 * 0.75 = 33.0)
      TOTAL:         80 -> 60 train + 20 held_out

    At 3000 new R2a rows / 60 unique train = 50x multiplicity (close to
    R1b1's 54x; well above lowmult's 20x memorization regime).

    Cross-rung invariant: R2a emits `A plus B` with B in [2,9]. R0/R1/R1b1/
    R1b2 all have B in {0, 1}; R2 (diagnosis-only) emits A_plus_B over
    the SAME range with 80/20 split -> overlaps R2a by construction.
    R2 stays excluded from build_rung_splits default + auto-excluded by
    DIAGNOSIS_ONLY_RUNGS in replay.py.
    """
    train_set: set = set()
    held_out_set: set = set()

    # Build (A, B) tuples + classify by phenomenon (addition only)
    by_phenom: dict[str, list[tuple[int, int]]] = {}
    for A in range(10, 20):
        for B in range(2, 10):
            phenom = _r2a_phenomenon(A, B)
            by_phenom.setdefault(phenom, []).append((A, B))

    # Stratify 75/25 per phenomenon with phenom-distinct _stable_seed
    for phenom, pairs in by_phenom.items():
        rng = random.Random(_stable_seed("R2a_partition", seed, phenom))
        rng.shuffle(pairs)
        split = int(len(pairs) * train_frac)
        train_set.update(pairs[:split])
        held_out_set.update(pairs[split:])
    return train_set, held_out_set


R2_TEMPLATES = ("A_plus_B", "A_minus_B")
R2_PHENOMENA = ("plus_no_carry", "plus_carry", "minus_no_borrow", "minus_borrow")


def _r2_phenomenon(template: str, A: int, B: int) -> str:
    """Classify (template, A, B) into one of R2's 4 phenomena.

    Per codex msg 1779476750248-2dca0aa7: stratify by phenomenon so train
    AND held_out both contain carry AND borrow cases.
    """
    if template == "A_plus_B":
        return "plus_carry" if (A % 10) + B >= 10 else "plus_no_carry"
    if template == "A_minus_B":
        return "minus_borrow" if (A % 10) < B else "minus_no_borrow"
    raise ValueError(f"unknown R2 template: {template!r}")


def _enumerate_partition_r2(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R2's teens variable-B
    operand space (codex msg 1779476750248-2dca0aa7 after R1b2 v2 replay50
    PASS at c2686cc).

    R2 is the smallest TRUE multi-digit ± bridge: introduces a real
    second operand B (not literal 1) while keeping A in teens [10,19].
    Output stays in [1, 28] (no 3-digit class). Stratifies by
    PHENOMENON, not just A bucket — guarantees carry/borrow examples
    in both splits.

    Pool: A in [10, 19] (10 vals) × B in [2, 9] (8 vals) × 2 templates
    = 160 (template, A, B) tuples total. 80/20 floor split per
    (template, phenomenon) bucket. Phenomenon counts:
      plus_no_carry: 36 -> 28 train + 8 held
      plus_carry:    44 -> 35 train + 9 held
      minus_no_borrow: 36 -> 28 train + 8 held
      minus_borrow:  44 -> 35 train + 9 held
      TOTAL:         160 -> 126 train + 34 held_out

    Cross-rung-train invariant: R2 emits `A plus B` / `A minus B` with
    B in [2,9]. R0 has no second operand. R1 has B=0 (additive side).
    R1b1 has B=1 (plus). R1b2 has B=1 (minus). All disjoint by B-value
    in question text. R1b/R1b2 templates with B=1 over A in [1,99]
    DO NOT overlap because R2's B never equals 1.
    """
    train_set: set = set()
    held_out_set: set = set()

    # Build all (template, A, B) tuples + classify by phenomenon
    by_bucket: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for template in R2_TEMPLATES:
        for A in range(10, 20):
            for B in range(2, 10):
                phenom = _r2_phenomenon(template, A, B)
                bucket = (template, phenom)
                by_bucket.setdefault(bucket, []).append((template, A, B))

    # Stratify 80/20 per (template, phenomenon) bucket with bucket-distinct
    # _stable_seed so different phenomena don't synchronize their shuffles
    for (template, phenom), pairs in by_bucket.items():
        rng = random.Random(_stable_seed("R2_partition", seed, template, phenom))
        rng.shuffle(pairs)
        split = int(len(pairs) * train_frac)
        train_set.update(pairs[:split])
        held_out_set.update(pairs[split:])
    return train_set, held_out_set


def _gen_r1b3(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b3 constant K=2 addition (codex msg 1779479973262-6d7445d2 after
    R2a v1 failed 0.045 at 558fcc1; variable-B reframed as blocker).

    Single template `what is A plus 2?` -> A+2. A in [1, 97]
    bucket-stratified. Output [3, 99]; no 3-digit class. Extends the
    locked constant-B single-template pattern (R1b1 K=1, R1b2 K=-1)
    to K=2 before attempting variable-B again.

    Train pool = 77 integers A; held_out pool = 20."""
    train_pool, held_out_pool = _enumerate_partition_r1b3(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A = rng.choice(pool_list)
        q = f"what is {A} plus 2?"
        expected = A + 2
        out.append({"question": q, "expected": expected, "rung": "R1b3"})
    return out


def _gen_r2a(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R2a teens addition-only (codex msg 1779478819906-0e30503e after
    full R2 failed v1+v2 n_train=8000 at c2f4f8d).

    Single template `what is A plus B?` -> A+B. A in [10,19], B in [2,9],
    phenomenon-stratified 75/25 across plus_no_carry + plus_carry.
    Output [12, 28]; no 3-digit class. Disjoint from R0/R1/R1b1/R1b2
    (B in [2,9] vs their B in {0,1}); overlaps R2 A_plus_B rows by
    construction (R2 diagnosis-only).

    Train pool = 60 unique (A, B); held_out pool = 20."""
    train_pool, held_out_pool = _enumerate_partition_r2a(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        A, B = rng.choice(pool_list)
        q = f"what is {A} plus {B}?"
        expected = A + B
        out.append({"question": q, "expected": expected, "rung": "R2a"})
    return out


def _gen_r2(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R2 teens variable-B (codex msg 1779476750248-2dca0aa7 after R1b2
    v2 replay50 PASS at c2686cc).

    Two templates × A in [10,19] × B in [2,9], stratified by phenomenon
    (plus_no_carry, plus_carry, minus_no_borrow, minus_borrow). Output
    in [1, 28]; no 3-digit class. Designed disjoint from R0/R1/R1b1/R1b2
    (B never equals 0 or 1).

    Train pool = 126 unique (template, A, B) tuples; held_out pool = 34.
    """
    train_pool, held_out_pool = _enumerate_partition_r2(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        template, A, B = rng.choice(pool_list)
        if template == "A_plus_B":
            q = f"what is {A} plus {B}?"
            expected = A + B
        elif template == "A_minus_B":
            q = f"what is {A} minus {B}?"
            expected = A - B
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown R2 template: {template!r}")
        out.append({"question": q, "expected": expected, "rung": "R2"})
    return out


def _gen_r3(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R3: multiplication-only on [0,9]² with 17×23 force-injected into
    held_out. NO division per codex spec (formatting/remainder ambiguity).
    Train pool = 80 unique (A, B) pairs; held_out pool = 20 + (17, 23)."""
    train_pool, held_out_pool = _enumerate_partition_r3(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order
    out = []
    while len(out) < n:
        A, B = rng.choice(pool_list)
        expected = A * B
        q = f"what is {A} times {B}?"
        out.append({"question": q, "expected": expected, "rung": "R3"})
    return out


def _gen_r4(rng: random.Random, spec: dict, n: int) -> list[dict]:
    # Multi-step compound: (A op B) op C with no division
    a_lo, a_hi = spec["A_range"]
    b_lo, b_hi = spec["B_range"]
    c_lo, c_hi = spec["C_range"]
    out = []
    while len(out) < n:
        A = rng.randint(a_lo, a_hi)
        B = rng.randint(b_lo, b_hi)
        C = rng.randint(c_lo, c_hi)
        # Mix +, -, ×; honor operator precedence (× before +/-)
        op1 = rng.choice(["plus", "minus", "times"])
        op2 = rng.choice(["plus", "minus", "times"])
        # Evaluate respecting precedence
        # Convert to Python and eval safely (small ints, no float)
        py_op = {"plus": "+", "minus": "-", "times": "*"}
        expr = f"{A} {py_op[op1]} {B} {py_op[op2]} {C}"
        # Restrict to int-safe arithmetic
        expected = eval(expr, {"__builtins__": {}})
        q = f"what is {A} {op1} {B} {op2} {C}?"
        out.append({"question": q, "expected": expected, "rung": "R4"})
    return out


def _gen_r5(rng: random.Random, n: int, split: str) -> list[dict]:
    """Paraphrase rung — same math primitives as R0-R4 but varied phrasing.
    train: 3 phrasings; held_out: 3 different phrasings."""
    phrasings = _R5_PHRASINGS_TRAIN if split == "train" else _R5_PHRASINGS_HELDOUT
    out = []
    while len(out) < n:
        # Choose primitive
        primitive_rung = rng.choice(["R1", "R3"])  # single-digit ± or ×
        A = rng.randint(0, 9)
        B = rng.randint(0, 9)
        if primitive_rung == "R1":
            op = rng.choice(["+", "-"])
            expr = f"{A} {op} {B}"
            expected = A + B if op == "+" else A - B
        else:
            expr = f"{A} * {B}"
            expected = A * B
        template = rng.choice(phrasings)
        q = template.format(expr=expr)
        out.append({"question": q, "expected": expected, "rung": "R5"})
    return out


def _gen_r6(rng: random.Random, n: int, split: str) -> list[dict]:
    """Templated word problem rung. Train and held_out use disjoint
    template banks (with same math operations) so the model must transfer
    primitives across novel surface form."""
    templates = _R6_TEMPLATES_TRAIN if split == "train" else _R6_TEMPLATES_HELDOUT
    out = []
    while len(out) < n:
        template, fn = rng.choice(templates)
        A = rng.randint(1, 20)
        B = rng.randint(1, 20)
        # For subtraction templates, ensure A >= B (so result non-negative
        # for the word-problem framing)
        if "checked out" in template or "eaten" in template:
            A = max(A, B)
        expected = fn(A, B)
        q = template.format(A=A, B=B)
        out.append({"question": q, "expected": expected, "rung": "R6"})
    return out


def make_rung_examples(
    rung: str,
    n: int,
    *,
    seed: int,
    split: Literal["train", "held_out"],
) -> list[dict]:
    """Deterministic synthetic examples for a single rung.

    Returns list of dicts: {"question": str, "expected": int, "rung": str}.

    R7 (GSM8k) is NOT generated here — it's served from
    scripts.train_hrm_text_158.load_gsm8k_splits directly.
    """
    if rung == "R7":
        raise ValueError(
            "R7 is GSM8k; load from load_gsm8k_splits() rather than make_rung_examples"
        )
    if rung not in RUNG_NAMES:
        raise ValueError(f"unknown rung {rung!r}; valid: {RUNG_NAMES}")
    if split not in ("train", "held_out"):
        raise ValueError(f"split must be 'train' or 'held_out'; got {split!r}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    rng = _make_rng(rung, seed, split)
    if rung == "R0":
        return _gen_r0(rng, _RUNG_SPEC["R0"][split], n, seed=seed, split=split)
    if rung == "R1":
        return _gen_r1(rng, _RUNG_SPEC["R1"][split], n, seed=seed, split=split)
    if rung == "R1b1":
        return _gen_r1b1(rng, _RUNG_SPEC["R1b1"][split], n, seed=seed, split=split)
    if rung == "R1b2a":
        return _gen_r1b2a(rng, _RUNG_SPEC["R1b2a"][split], n, seed=seed, split=split)
    if rung == "R1b2":
        return _gen_r1b2(rng, _RUNG_SPEC["R1b2"][split], n, seed=seed, split=split)
    if rung == "R1b3":
        return _gen_r1b3(rng, _RUNG_SPEC["R1b3"][split], n, seed=seed, split=split)
    if rung == "R1b4":
        return _gen_r1b4(rng, _RUNG_SPEC["R1b4"][split], n, seed=seed, split=split)
    if rung == "R1b4v2":
        return _gen_r1b4v2(rng, _RUNG_SPEC["R1b4v2"][split], n, seed=seed, split=split)
    if rung == "R1b5":
        return _gen_r1b5(rng, _RUNG_SPEC["R1b5"][split], n, seed=seed, split=split)
    if rung == "R1b6":
        return _gen_r1b6(rng, _RUNG_SPEC["R1b6"][split], n, seed=seed, split=split)
    if rung == "R1b7":
        return _gen_r1b7(rng, _RUNG_SPEC["R1b7"][split], n, seed=seed, split=split)
    if rung == "R1b8":
        return _gen_r1b8(rng, _RUNG_SPEC["R1b8"][split], n, seed=seed, split=split)
    if rung == "R1b":
        return _gen_r1b(rng, _RUNG_SPEC["R1b"][split], n, seed=seed, split=split)
    if rung == "R2a":
        return _gen_r2a(rng, _RUNG_SPEC["R2a"][split], n, seed=seed, split=split)
    if rung == "R2":
        return _gen_r2(rng, _RUNG_SPEC["R2"][split], n, seed=seed, split=split)
    if rung == "R3":
        return _gen_r3(rng, _RUNG_SPEC["R3"][split], n, seed=seed, split=split)
    if rung == "R4":
        return _gen_r4(rng, _RUNG_SPEC["R4"][split], n)
    if rung == "R5":
        return _gen_r5(rng, n, split)
    if rung == "R6":
        return _gen_r6(rng, n, split)
    raise NotImplementedError(f"generator for {rung!r} not implemented")
