#!/bin/bash
# Phase-2 A/B: per-surface bank-gate audits (target acquire + retained curriculum
# + close sibling + language), both arms x 6 saves. One flag per invocation
# (probe modes are mutually exclusive).
set -uo pipefail
cd ~/claw-code-hrm-158
source ~/hrm158-venv/bin/activate
OUT=~/audit_out/rotor3b_attempt10
mkdir -p "$OUT"

SURFACES=(
  "trace_train:--l0c2k2-addition-60to89-trace-train-audit:--max-gen 160"
  "trace_held:--l0c2k2-addition-60to89-trace-held-audit:--max-gen 160"
  "add50s:--l0c2k2-addition-50s-audit:"
  "add120:--l0c2k2-addition-120-audit:"
  "add120k5to8:--l0c2k2-addition-120-k5to8-audit:"
  "idfull:--l0c2k1-identity-full-audit:"
  "l0c1:--l0c1-audit:"
  "language:--language-supports:"
)

for ck in calm/hrm/checkpoints/hrm_text_158_phase3_L0c2K2add60to89TraceRotor3b_*_final_step0*.pt \
          calm/hrm/checkpoints/hrm_text_158_phase3_L0c2K2add60to89Trace_seed0017_*_final_step0*.pt; do
  name=$(basename "$ck" .pt)
  short=$(echo "$name" | grep -oE 'step0[0-9]+' | tail -1)
  arm="fp_control"; echo "$name" | grep -q Rotor3b && arm="rotor3b"
  for entry in "${SURFACES[@]}"; do
    sname="${entry%%:*}"
    rest="${entry#*:}"
    flag="${rest%%:*}"
    extra="${rest#*:}"
    out="$OUT/${arm}_${short}_${sname}_audit.json"
    if [ -s "$out" ]; then echo "AUDIT_SKIP ${arm} ${short} ${sname}"; continue; fi
    echo "AUDIT_START ${arm} ${short} ${sname} $(date -u +%H:%M:%S)"
    PYTHONPATH=. python -u scripts/probe_hrm_text_158.py \
      --ckpt-path "$ck" \
      $flag $extra \
      --audit-output-json "$out" \
      --use-cached-ternary-infer --use-kv-cache-decode \
      --use-batched-probe-eval --probe-batch-size 32 \
      > /tmp/last_audit.log 2>&1 \
      && echo "AUDIT_OK ${arm} ${short} ${sname}" \
      || { echo "AUDIT_FAIL ${arm} ${short} ${sname}"; tail -5 /tmp/last_audit.log; }
  done
done
echo "ALL_PHASE2_AUDITS_DONE"
