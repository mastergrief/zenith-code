#!/bin/bash
# rotor3b attempt-10 terminal A/B: exhaustive A0 audits, both arms x 6 saves, on the 1070 box.
set -uo pipefail
cd ~/claw-code-hrm-158
source ~/hrm158-venv/bin/activate
mkdir -p ~/audit_out/rotor3b_attempt10
for ck in calm/hrm/checkpoints/hrm_text_158_phase3_L0c2K2add60to89TraceRotor3b_*_final_step0*.pt \
          calm/hrm/checkpoints/hrm_text_158_phase3_L0c2K2add60to89Trace_seed0017_*_final_step0*.pt; do
  name=$(basename "$ck" .pt)
  # Names carry TWO step tokens (parent step + save step) — take the LAST.
  short=$(echo "$name" | grep -oE 'step0[0-9]+' | tail -1)
  arm="fp_control"; echo "$name" | grep -q Rotor3b && arm="rotor3b"
  out=~/audit_out/rotor3b_attempt10/${arm}_${short}_audit.json
  if [ -s "$out" ]; then echo "AUDIT_SKIP ${arm} ${short} (exists)"; continue; fi
  echo "AUDIT_START ${arm} ${short} $(date -u +%H:%M:%S)"
  PYTHONPATH=. python -u scripts/probe_hrm_text_158.py \
    --ckpt-path "$ck" \
    --exhaustive-finite-supports \
    --audit-output-json "$out" \
    --use-cached-ternary-infer --use-kv-cache-decode \
    --use-batched-probe-eval --probe-batch-size 32 \
    && echo "AUDIT_OK ${arm} ${short}" || echo "AUDIT_FAIL ${arm} ${short}"
done
echo "ALL_AUDITS_DONE"
