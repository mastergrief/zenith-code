# ERRATA — phase4a_kv_ternary_screen_receipt.json

Gate-2 audit finding (co_lead review msg `1784793102933-64c5698f`, post-hoc
audit of commit `52de311`): the frozen receipt's `bits_ledger` keys are
**mislabeled**. The keys read `turbo2` / `turbo3` but the VALUES are the
**ternary** ledgers:

- `"turbo2"` in this receipt is actually **ternary + fp16 scale**
  (code_bits=1.625, bpw=1.75, sub-2 TRUE).
- `"turbo3"` in this receipt is actually **ternary + int8 scale**
  (bpw=1.6875, sub-2 TRUE).

The TRUE turbo2 ledger is 2.125 bpw scale-inclusive and is **NOT sub-2**.
Do not read this receipt as evidence that 2-bit/turbo2 codes clear the
sub-2 bar — they cannot (4-level codes are 2.0 bpw before scales).

Root cause: `hrm_text_158_rotor_forward_activation_screen.py` emitted fixed
key names for every surface. Fixed post-audit to emit
`ternary_fp16_scale` / `ternary_int8_scale` for `--surface kv_ternary`.
The frozen receipt is left byte-unchanged (frozen artifacts are immutable);
this sidecar is the authoritative label correction.

Numeric values, prereg gates, and the sub-2 verdict itself were
independently replayed and CONFIRMED in the same audit (exact drop 0,
d_CE ~= 0.0064 < 0.1; 208/128 + 16 = 1.75 bpw; +8 int8 = 1.6875).
