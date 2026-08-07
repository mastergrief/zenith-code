set -euo pipefail
# External control-plane log (OUTSIDE frozen ROOT). Required env.
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S1" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
: "${R1L_ROOT:?}"
python3 - <<'PY'
import hashlib, json, os, shutil, subprocess, importlib.util, sys
from pathlib import Path
from datetime import datetime, timezone

REPO = Path('/mnt/c/Users/gabes/projects/claw-code-hrm-text-158')
ROOT = Path(os.environ['R1L_ROOT'])
HEAD = '0636177fbe52d8c6ff5db71312f51240b31fceb2'
R1_CPU = '717f6346324388f83126763769c30b9bad53dc45'
W6_SRC = Path('/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt')
W6_SHA = '9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec'
R3_FREEZE = Path('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/p1_freeze_manifest.json')
R3_RECEIPT = Path('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/p1b_live_conversion_receipt_0636177f.json')
R3_BINDING = Path('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/data_binding_manifest.json')
R3_FREEZE_SHA = 'b63d9ff44ca12554834f72555a62800837302bfedc51e15adac66fb7653f2305'
R3_RECEIPT_SHA = 'ea69cf4750336ea2914ee1f001b9c48a91c49a8c190ed5e4ddde6b36322240e6'
R3_BINDING_SHA = '6edcd73bbe6f91f20c77f524e88888502cfac95e9ffe2745e40add52011317f1'

code_dir = ROOT / 'code'
assert code_dir.is_dir()
for p in code_dir.iterdir():
    if p.name != '.keep':
        raise SystemExit(f'code_dir_not_empty:{p}')
subprocess.check_call(
    f"git -C {REPO} archive --format=tar {HEAD} | tar -C {code_dir} -xf -",
    shell=True,
)

ar = code_dir / 'calm/hrm_text_158/native_full_stack/activation_relief.py'
assert ar.is_file(), ar
spec = importlib.util.spec_from_file_location('ar_s1_canon_v13', ar)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
_canonical_json_dumps = mod._canonical_json_dumps
compute_launch_manifest_sha256 = mod.compute_launch_manifest_sha256
compute_proof_env_hash_sha256 = mod.compute_proof_env_hash_sha256

def dir_sha(path: Path) -> str:
    lines = []
    for p in sorted(path.rglob('*')):
        if p.is_file() and not p.is_symlink():
            rel = p.relative_to(path).as_posix()
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {rel}")
        elif p.is_symlink():
            rel = p.relative_to(path).as_posix()
            lines.append(f"symlink:{os.readlink(p)}  {rel}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()

clean_run_dir_sha256 = dir_sha(code_dir)
w6_dst = ROOT / 'artifacts' / 'w6_parent_readonly.pt'
assert not w6_dst.exists()
shutil.copy2(W6_SRC, w6_dst)
os.chmod(w6_dst, 0o444)
assert hashlib.sha256(w6_dst.read_bytes()).hexdigest() == W6_SHA

auth = {
  'schema': 'r1l_p1_authority_binding/v13',
  'head_sha': HEAD,
  'r1_cpu_base_commit_sha': R1_CPU,
  'p1_root': '/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3',
  'p1_freeze_manifest_sha256': R3_FREEZE_SHA,
  'p1_receipt_sha256': R3_RECEIPT_SHA,
  'data_binding_manifest_sha256': R3_BINDING_SHA,
  'p1_freeze_manifest_path': str(R3_FREEZE),
  'p1_receipt_path': str(R3_RECEIPT),
  'data_binding_manifest_path': str(R3_BINDING),
  'excluded_prior_roots': [
    '/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r2',
    '/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f',
    '/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r1',
  ],
  'stale_prior_r1l_path': '/home/gabe/claw-code-creditdir/transient_fp_credit/r1l_launch_20260619T073241Z',
  'stale_prior_r1l_status': 'historical_reference_only_not_authority',
  'w6_sha256': W6_SHA,
  'clean_run_dir_sha256': clean_run_dir_sha256,
}
auth_path = ROOT / 'authority_binding.json'
fd = os.open(str(auth_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, 'w') as f:
    f.write(json.dumps(auth, indent=2, sort_keys=True) + '\n')

utc = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
manifest_map = {
  'r1_cpu_base_commit_sha': R1_CPU,
  'launch_source_commit_sha': HEAD,
  'archive_created_at_utc': utc,
  'archive_method': 'git_archive_HEAD',
  'p1_freeze_manifest_sha256': R3_FREEZE_SHA,
  'p1_receipt_sha256': R3_RECEIPT_SHA,
  'data_binding_manifest_sha256': R3_BINDING_SHA,
  'p1_receipt_json': str(R3_RECEIPT),
  'authority_binding_sha256': hashlib.sha256(auth_path.read_bytes()).hexdigest(),
  'clean_run_dir_sha256': clean_run_dir_sha256,
  'w6_parent_sha256': W6_SHA,
}
manifest_map = {str(k): str(v) for k, v in manifest_map.items()}
canon_text = _canonical_json_dumps(manifest_map)
man_path = ROOT / 'launch_manifest.json'
fd = os.open(str(man_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, 'wb') as f:
    f.write(canon_text.encode('utf-8'))
file_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()
canon_sha = compute_launch_manifest_sha256(manifest_map)
assert file_sha == canon_sha, ('manifest_canonical_mismatch', file_sha, canon_sha)
print('S1_MANIFEST_CANONICAL_OK', file_sha)

trainer_log = ROOT / 'logs' / 'launch.log'
if not trainer_log.exists():
    trainer_log.write_bytes(b'')

env_map = {
  'PYTHONDONTWRITEBYTECODE': '1',
  'PYTHONPATH': str(code_dir),
  'CUDA_VISIBLE_DEVICES': '0',
  'HF_HUB_OFFLINE': '1',
  'HF_DATASETS_OFFLINE': '1',
  'TRANSFORMERS_OFFLINE': '1',
  'R1L_LAUNCH_RECEIPT_JSON': str(ROOT / 'receipts' / 'r1l_launch_runtime_receipt.json'),
  'R1L_LAUNCH_LOG': str(trainer_log),
  'R1L_W6_PARENT_PATH': str(w6_dst),
  'R1L_LAUNCH_MANIFEST_JSON': str(man_path),
  'R1L_CLEAN_RUN_DIR_SHA256': clean_run_dir_sha256,
  'R1L_ANCESTRY_VERIFIED': '1',
  'TORCH_CUDA_ALLOC_CONF': '',
  'CUBLAS_WORKSPACE_CONFIG': '',
  'P1_RECEIPT_JSON': str(R3_RECEIPT),
  'P1B_LIVE_CONVERSION_RECEIPT_JSON': str(R3_RECEIPT),
}
env_map = {str(k): str(v) for k, v in env_map.items()}
env_canon = _canonical_json_dumps(env_map)
env_path = ROOT / 'launch_env.json'
fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, 'wb') as f:
    f.write(env_canon.encode('utf-8'))
proof_env_hash = compute_proof_env_hash_sha256(env_map)
env_from_file = {str(k): str(v) for k, v in json.loads(env_path.read_text()).items()}
assert compute_proof_env_hash_sha256(env_from_file) == proof_env_hash
# Also: file bytes of the PROOF_ENV subset serialization equals hash input reconstruction
print('S1_PROOF_ENV_HASH_OK', proof_env_hash)

print('S1_ARCHIVE_W6_BIND_MANIFEST_ENV_OK')
print('CLEAN_RUN_DIR_SHA256', clean_run_dir_sha256)
print('MANIFEST_SHA256', file_sha)
print('ENV_SHA256', hashlib.sha256(env_path.read_bytes()).hexdigest())
print('AUTH_SHA256', hashlib.sha256(auth_path.read_bytes()).hexdigest())
print('W6_DST_SHA256', hashlib.sha256(w6_dst.read_bytes()).hexdigest())
print('TRAINER_LOG_PATH', str(trainer_log))
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S1 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S1 rc=$rc" | tee -a "$LOG" >/dev/null; fi
exit $rc
