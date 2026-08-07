set -euo pipefail
# External control-plane log (OUTSIDE frozen ROOT). Required env.
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S3" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
: "${R1L_ROOT:?}"
: "${R1L_RUNNER_LOG:?}"
python3 - <<'PY'
import hashlib, json, re, importlib.util, sys, os
from pathlib import Path

ROOT = Path(os.environ['R1L_ROOT'])
RUNNER_LOG = Path(os.environ['R1L_RUNNER_LOG'])
HEAD = '0636177fbe52d8c6ff5db71312f51240b31fceb2'
R1_CPU = '717f6346324388f83126763769c30b9bad53dc45'
W6_SHA = '9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec'
R3_FREEZE_SHA = 'b63d9ff44ca12554834f72555a62800837302bfedc51e15adac66fb7653f2305'
R3_RECEIPT_SHA = 'ea69cf4750336ea2914ee1f001b9c48a91c49a8c190ed5e4ddde6b36322240e6'
R3_BINDING_SHA = '6edcd73bbe6f91f20c77f524e88888502cfac95e9ffe2745e40add52011317f1'
ARGV_FROM_SCRIPT = ["/home/gabe/claw-code-creditdir/transient_fp_credit/r1l_launch_HEAD_0636177f_r1/code/scripts/train_hrm_text_158.py", "--load-from", "/home/gabe/claw-code-creditdir/transient_fp_credit/r1l_launch_HEAD_0636177f_r1/artifacts/w6_parent_readonly.pt", "--use-ternary-bulk", "--activation-relief-lossless-recompute-launch-proof", "--epochs", "1", "--n-train-cap", "8", "--batch-size", "8", "--max-len", "384", "--hidden-size", "512", "--n-layers", "8", "--num-heads", "4", "--expansion", "4", "--H-cycles", "2", "--L-cycles", "3", "--parent-consistency-weight", "1.0", "--retained-support", "L0b:0.01", "--retained-support-batch", "8", "--curriculum-rung", "R0", "--use-broad-tokenizer", "--seed", "17"]
AR_PATH = ROOT / 'code' / 'calm/hrm_text_158/native_full_stack/activation_relief.py'

def append_runner(line: str) -> None:
    RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNNER_LOG.open('a') as f:
        f.write(line.rstrip() + '\n')
        f.flush()

receipt_path = ROOT/'receipts'/'r1l_launch_runtime_receipt.json'
assert receipt_path.is_file(), 'receipt_missing'
receipt_dict = json.loads(receipt_path.read_text())
auth = json.loads((ROOT/'authority_binding.json').read_text())
man = json.loads((ROOT/'launch_manifest.json').read_text())
man_bytes = (ROOT/'launch_manifest.json').read_bytes()
env_bytes = (ROOT/'launch_env.json').read_bytes()

# Correct artifact set: launch_log_at_mint.log (producer activation_relief.py:1710-1731)
snap = receipt_path.parent / 'launch_log_at_mint.log'
assert snap.is_file() and snap.stat().st_size > 0, 'launch_log_at_mint_missing'
log_bytes = snap.read_bytes()

# proof_command_argv exact equality vs frozen[2:]
got_argv = list(receipt_dict.get('proof_command_argv') or [])
if got_argv != list(ARGV_FROM_SCRIPT):
    append_runner('R1L_TERMINAL_FAIL')
    append_runner('ARGV_MISMATCH')
    print('R1L_TERMINAL_FAIL')
    print('ARGV_MISMATCH', got_argv[:3], 'vs', list(ARGV_FROM_SCRIPT)[:3])
    raise SystemExit(1)
print('S3_ARGV_EQUALITY_OK', len(got_argv))

# Canonical validators
assert AR_PATH.is_file(), ('activation_relief_missing', str(AR_PATH))
spec = importlib.util.spec_from_file_location('activation_relief_canonical_r1l_v13', AR_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
try:
    receipt_obj = mod.launch_runtime_backward_receipt_from_dict(receipt_dict)
    mod.validate_launch_runtime_backward_receipt(receipt_obj)
    mod.validate_launch_runtime_backward_artifacts(
        receipt_obj,
        launch_manifest_bytes=man_bytes,
        env_snapshot_bytes=env_bytes,
        log_bytes=log_bytes,  # launch_log_at_mint snapshot bytes
    )
except Exception as exc:
    append_runner('R1L_TERMINAL_FAIL')
    append_runner('CANONICAL_VALIDATOR_EXCEPTION ' + type(exc).__name__ + ': ' + str(exc))
    print('R1L_TERMINAL_FAIL')
    print('CANONICAL_VALIDATOR_EXCEPTION', type(exc).__name__, str(exc))
    raise SystemExit(1)
print('S3_CANONICAL_VALIDATORS_OK')

assert receipt_obj.live_readiness_row_flip_authorized is True
assert tuple(receipt_obj.readiness_row_flip_authorized_surface_names) == (
    'backward_saved_tensors_transients',
)
# log sha binding
assert hashlib.sha256(log_bytes).hexdigest() == receipt_dict.get('log_artifact_sha256')

# additional manual field asserts
fails = []
def req(cond, msg):
    if not cond:
        fails.append(msg)
req(receipt_dict.get('launch_runtime_validation_pass') is True, 'launch_runtime_validation_pass')
req(receipt_dict.get('schema_version') == 'hrm_text_158_r1_backward_launch/v1.gpu_runtime_validation', 'schema')
req(receipt_dict.get('target_name') == 'r1_backward_saved_tensors_launch_runtime', 'target')
req(receipt_dict.get('launch_source_commit_sha') == HEAD, 'launch_source')
req(receipt_dict.get('r1_cpu_base_commit_sha') == R1_CPU, 'r1_cpu_base')
req(receipt_dict.get('w6_parent_sha256_before') == W6_SHA, 'w6_before')
req(receipt_dict.get('w6_parent_sha256_after') == W6_SHA, 'w6_after')
req(receipt_dict.get('applier_result_ready_for_main_science') is False, 'ready_main_applier')
req(receipt_dict.get('ready_for_main_science', False) is False, 'ready_main')
req(receipt_dict.get('loss_finite_main') is True, 'loss_main')
req(receipt_dict.get('loss_finite_retained') is True, 'loss_retained')
req(receipt_dict.get('main_path_proven') is True, 'main_path')
req(receipt_dict.get('main_recompute_checkpoint_fired') is True, 'recompute_fired')
req(receipt_dict.get('ancestry_verified_at_launch_preflight') is True, 'ancestry')
pb = receipt_dict.get('proof_batch_digest_sha256', '')
req(isinstance(pb, str) and re.fullmatch(r'[0-9a-f]{64}', pb) is not None, 'proof_batch_digest')
req(auth['p1_freeze_manifest_sha256'] == R3_FREEZE_SHA, 'auth_freeze')
req(auth['p1_receipt_sha256'] == R3_RECEIPT_SHA, 'auth_receipt')
req(auth['data_binding_manifest_sha256'] == R3_BINDING_SHA, 'auth_binding')
req(auth['head_sha'] == HEAD, 'auth_head')
req(man['launch_source_commit_sha'] == HEAD, 'man_head')
req(man['p1_freeze_manifest_sha256'] == R3_FREEZE_SHA, 'man_freeze')
blob = json.dumps(receipt_dict) + json.dumps(auth) + json.dumps(man)
req('4f6bee6f30ba1165a1533b18a5759d893f4098997e541507687aec4c8c6950f1' not in blob, 'stale_p1_sha')
surfaces = set(receipt_dict.get('readiness_row_flip_authorized_surface_names') or [])
req(surfaces.issubset({'backward_saved_tensors_transients'}), ('surfaces', surfaces))
w6 = ROOT/'artifacts'/'w6_parent_readonly.pt'
req(hashlib.sha256(w6.read_bytes()).hexdigest() == W6_SHA, 'w6_file')

if fails:
    append_runner('R1L_TERMINAL_FAIL')
    append_runner('FAILS ' + json.dumps(fails, default=str))
    print('R1L_TERMINAL_FAIL')
    print('FAILS', fails)
    raise SystemExit(1)
append_runner('R1L_TERMINAL_PASS')
print('S3_TERMINAL_PASS_APPENDED')
print('PROOF_BATCH_DIGEST', pb)
print('RECEIPT_SHA256', hashlib.sha256(receipt_path.read_bytes()).hexdigest())
print('S3_LOG_ARTIFACT', 'receipts/launch_log_at_mint.log')
print('S3_STOP_MARKER_WRITER', 'S3_append_to_R1L_RUNNER_LOG')
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S3 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S3 rc=$rc" | tee -a "$LOG" >/dev/null; fi
exit $rc
