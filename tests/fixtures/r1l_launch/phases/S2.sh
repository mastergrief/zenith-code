set -euo pipefail
# External control-plane log (OUTSIDE frozen ROOT). Required env.
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S2" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
: "${R1L_ROOT:?}"
: "${R1L_LAUNCH_SOURCE_COMMIT_SHA:?R1L_LAUNCH_SOURCE_COMMIT_SHA required (40-hex launch source)}"
: "${R1L_S2_MODE:=gpu}"
python3 - <<'PY'
import hashlib, json, os, subprocess, sys, importlib.util
from pathlib import Path

ROOT = Path(os.environ['R1L_ROOT'])
MODE = os.environ.get('R1L_S2_MODE', 'gpu')
env = {str(k): str(v) for k, v in json.loads((ROOT/'launch_env.json').read_text()).items()}
child_env = os.environ.copy()
child_env.update(env)
trainer_log = Path(env['R1L_LAUNCH_LOG'])
assert str(trainer_log).startswith(str(ROOT)), 'trainer_log_must_be_under_ROOT'
trainer_log.parent.mkdir(parents=True, exist_ok=True)
if not trainer_log.exists():
    trainer_log.write_bytes(b'')

# Launch source commit — parameter (not a fixture literal). Distinct from P1-mint freeze head.
import re as _re_launch
_launch = os.environ.get('R1L_LAUNCH_SOURCE_COMMIT_SHA', '').strip()
assert _re_launch.fullmatch(r'[0-9a-f]{40}', _launch), (
    'R1L_LAUNCH_SOURCE_COMMIT_SHA_required_40hex', _launch,
)
HEAD = _launch  # launch_source only
R1_CPU = '717f6346324388f83126763769c30b9bad53dc45'
W6_SHA = '9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec'

# proof_command_argv authority derived from required R1L_ROOT — no launch-root literal
_w6_parent = str(env.get('R1L_W6_PARENT_PATH') or (ROOT / 'artifacts' / 'w6_parent_readonly.pt'))
assert str(_w6_parent).startswith(str(ROOT)), (
    'w6_parent_must_be_under_R1L_ROOT', _w6_parent, str(ROOT),
)
ARGV_FROM_SCRIPT = [
    str(ROOT / 'code' / 'scripts' / 'train_hrm_text_158.py'),
    '--load-from', _w6_parent,
    '--use-ternary-bulk',
    '--activation-relief-lossless-recompute-launch-proof',
    '--epochs', '1',
    '--n-train-cap', '8',
    '--batch-size', '8',
    '--max-len', '384',
    '--hidden-size', '512',
    '--n-layers', '8',
    '--num-heads', '4',
    '--expansion', '4',
    '--H-cycles', '2',
    '--L-cycles', '3',
    '--parent-consistency-weight', '1.0',
    '--retained-support', 'L0b:0.01',
    '--retained-support-batch', '8',
    '--curriculum-rung', 'R0',
    '--use-broad-tokenizer',
    '--seed', '17',
]
assert all('r1l_launch_HEAD_' not in str(a) for a in ARGV_FROM_SCRIPT), (
    'launch_root_literal_forbidden_in_argv', ARGV_FROM_SCRIPT[:3],
)

code = Path(env['PYTHONPATH'])
receipt_path = Path(env['R1L_LAUNCH_RECEIPT_JSON'])
assert not receipt_path.exists(), 'receipt_preexists'

ar = code / 'calm/hrm_text_158/native_full_stack/activation_relief.py'
spec = importlib.util.spec_from_file_location('ar_s2_v13', ar)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

if MODE == 'gpu':
    argv = [
      sys.executable, '-u', str(code/'scripts'/'train_hrm_text_158.py'),
      '--load-from', env['R1L_W6_PARENT_PATH'],
      '--use-ternary-bulk',
      '--activation-relief-lossless-recompute-launch-proof',
      '--epochs', '1',
      '--n-train-cap', '8',
      '--batch-size', '8',
      '--max-len', '384',
      '--hidden-size', '512',
      '--n-layers', '8',
      '--num-heads', '4',
      '--expansion', '4',
      '--H-cycles', '2',
      '--L-cycles', '3',
      '--parent-consistency-weight', '1.0',
      '--retained-support', 'L0b:0.01',
      '--retained-support-batch', '8',
      '--curriculum-rung', 'R0',
      '--use-broad-tokenizer',
      '--seed', '17',
    ]
    with trainer_log.open('ab') as log:
        log.write(('S2_GPU_LAUNCH_ARGV ' + json.dumps(argv) + '\n').encode())
        log.flush()
        proc = subprocess.run(
            argv, cwd=str(code), env=child_env, stdout=log, stderr=subprocess.STDOUT,
        )
    print('S2_GPU_LAUNCH_RC', proc.returncode)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    assert receipt_path.is_file(), 'receipt_missing'
    snap = receipt_path.parent / 'launch_log_at_mint.log'
    assert snap.is_file() and snap.stat().st_size > 0, 'launch_log_at_mint_missing'
    print('S2_GPU_LAUNCH_PROOF_OK', receipt_path)
    print('S2_LAUNCH_LOG_AT_MINT', snap)
elif MODE in ('synthetic', 'synthetic_fail'):
    with trainer_log.open('ab') as log:
        log.write(b'[hrm158] R1-L launch proof: synthetic fixture\n')
        log.write(('[hrm158] R1-L launch proof: receipt_json=' + str(receipt_path) + '\n').encode())
        log.flush()
    log_bytes = trainer_log.read_bytes()
    assert log_bytes, 'trainer_log_empty'
    log_sha = hashlib.sha256(log_bytes).hexdigest()
    snap = receipt_path.parent / 'launch_log_at_mint.log'
    assert not snap.exists()
    snap.write_bytes(log_bytes)

    man_bytes = (ROOT/'launch_manifest.json').read_bytes()
    manifest_map = {str(k): str(v) for k, v in json.loads(man_bytes.decode('utf-8')).items()}
    assert hashlib.sha256(man_bytes).hexdigest() == mod.compute_launch_manifest_sha256(manifest_map), (
        's2_manifest_file_not_canonical'
    )
    env_map = {str(k): str(v) for k, v in env.items()}
    env_bytes = (ROOT/'launch_env.json').read_bytes()

    # Build via source builder so launch_manifest_sha256/embedded come from source functions.
    kwargs = dict(
        launch_source_commit_sha=HEAD,
        launch_manifest_embedded=manifest_map,
        proof_env_embedded=env_map,
        proof_command_argv=list(ARGV_FROM_SCRIPT),
        clean_run_dir_sha256=env['R1L_CLEAN_RUN_DIR_SHA256'],
        w6_parent_path=env['R1L_W6_PARENT_PATH'],
        w6_parent_sha256=W6_SHA,
        gpu_name='synthetic',
        gpu_uuid='synthetic-uuid',
        driver_version='synthetic',
        cuda_version='synthetic',
        torch_version='synthetic',
        model_config_digest_sha256=hashlib.sha256(b'synthetic_model_cfg_v13').hexdigest(),
        proof_batch_digest_sha256=hashlib.sha256(b'synthetic_proof_batch_v13').hexdigest(),
        retained_support_digest_sha256=hashlib.sha256(b'synthetic_retained_v13').hexdigest(),
        main_baseline_saved_tensor_count=20,
        main_recompute_saved_tensor_count=10,
        main_saved_tensor_payload_bytes_baseline=1000,
        main_saved_tensor_payload_bytes_recompute=400,
        retained_side_in_scope=True,
        retained_side_baseline_saved_tensor_count=18,
        retained_side_recompute_saved_tensor_count=9,
        retained_saved_tensor_payload_bytes_delta=600,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=32 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
        log_artifact_sha256=log_sha,
        applier_base_surface_count_sub2=3,  # Type-1: no mint
        applier_result_sub2_surface_count=3,
        ancestry_verified_at_launch_preflight=True,
        r1_cpu_base_commit_sha=R1_CPU,
    )
    if MODE == 'synthetic_fail':
        # Legitimate failure: empty authorized surface via post-build tamper of payload
        # after builder — force live_readiness false by building then corrupting dict.
        receipt_obj = mod.build_launch_runtime_backward_validation_receipt(**kwargs)
        d = receipt_obj.to_dict()
        d['live_readiness_row_flip_authorized'] = False
        d['canonical_launch_artifact_sha256'] = None
        d['canonical_launch_artifact_sha256'] = mod.compute_canonical_launch_artifact_sha256(d)
        try:
            ro = mod.launch_runtime_backward_receipt_from_dict(d)
            mod.validate_launch_runtime_backward_receipt(ro)
            print('S2_SYNTHETIC_VALIDATED', True)
            raise SystemExit('expected_validation_failure_did_not_fire')
        except Exception as exc:
            print('S2_SYNTHETIC_VALIDATED', False)
            print('S2_SYNTHETIC_VALIDATE_FAIL', type(exc).__name__, str(exc)[:200])
            raise SystemExit(1)

    receipt_obj = mod.build_launch_runtime_backward_validation_receipt(**kwargs)
    try:
        mod.validate_launch_runtime_backward_receipt(receipt_obj)
        mod.validate_launch_runtime_backward_artifacts(
            receipt_obj,
            launch_manifest_bytes=man_bytes,
            env_snapshot_bytes=env_bytes,
            log_bytes=log_bytes,
        )
    except Exception as exc:
        print('S2_SYNTHETIC_VALIDATED', False)
        print('S2_SYNTHETIC_VALIDATE_FAIL', type(exc).__name__, str(exc)[:300])
        raise SystemExit(1)

    out = receipt_obj.to_dict()
    fd = os.open(str(receipt_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(out, indent=2, sort_keys=True) + '\n')
    disk = json.loads(receipt_path.read_text())
    ro = mod.launch_runtime_backward_receipt_from_dict(disk)
    mod.validate_launch_runtime_backward_receipt(ro)
    mod.validate_launch_runtime_backward_artifacts(
        ro,
        launch_manifest_bytes=man_bytes,
        env_snapshot_bytes=env_bytes,
        log_bytes=snap.read_bytes(),
    )
    print('S2_SYNTHETIC_LAUNCH_PROOF_OK', receipt_path)
    print('S2_LAUNCH_LOG_AT_MINT', snap)
    print('S2_SYNTHETIC_VALIDATED', True)
    print('S2_MODE', MODE)
    print('S2_MANIFEST_FILE_EQ_CANON', hashlib.sha256(man_bytes).hexdigest())
else:
    raise SystemExit(f'unknown_R1L_S2_MODE={MODE}')
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S2 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S2 rc=$rc" | tee -a "$LOG" >/dev/null; fi
exit $rc
