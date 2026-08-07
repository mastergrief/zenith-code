set -euo pipefail
# External control-plane log (OUTSIDE frozen ROOT). Required env.
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S0b" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
: "${R1L_ROOT:?}"
: "${R1L_LAUNCH_SOURCE_COMMIT_SHA:?R1L_LAUNCH_SOURCE_COMMIT_SHA required (40-hex launch source)}"
python3 - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path

P1_FREEZE_HEAD = '0636177fbe52d8c6ff5db71312f51240b31fceb2'  # P1-mint freeze head_sha identity (not launch)
# Launch source commit — parameter (not a fixture literal). Distinct from P1-mint freeze head.
import re as _re_launch
_launch = os.environ.get('R1L_LAUNCH_SOURCE_COMMIT_SHA', '').strip()
assert _re_launch.fullmatch(r'[0-9a-f]{40}', _launch), (
    'R1L_LAUNCH_SOURCE_COMMIT_SHA_required_40hex', _launch,
)
HEAD = _launch  # launch_source only
R1_CPU = '717f6346324388f83126763769c30b9bad53dc45'
W6_SRC = Path('/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt')
W6_SHA = '9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec'
REPO = '/mnt/c/Users/gabes/projects/claw-code-hrm-text-158'
pins = {
  'freeze': ('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/p1_freeze_manifest.json', 'b63d9ff44ca12554834f72555a62800837302bfedc51e15adac66fb7653f2305'),
  'receipt': ('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/p1b_live_conversion_receipt_0636177f.json', 'ea69cf4750336ea2914ee1f001b9c48a91c49a8c190ed5e4ddde6b36322240e6'),
  'binding': ('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3' + '/data_binding_manifest.json', '6edcd73bbe6f91f20c77f524e88888502cfac95e9ffe2745e40add52011317f1'),
}
R3 = Path('/home/gabe/claw-code-creditdir/transient_fp_credit/p1b_live_conversion_HEAD_0636177f_r3')
STALE = '/home/gabe/claw-code-creditdir/transient_fp_credit/r1l_launch_20260619T073241Z'

got = subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
assert got == HEAD, ('head_mismatch', got, HEAD)
r = subprocess.run(['git', '-C', REPO, 'merge-base', '--is-ancestor', R1_CPU, HEAD])
assert r.returncode == 0, 'ancestry_fail'
for name, (path, sha) in pins.items():
    p = Path(path)
    assert p.is_file() and not p.is_symlink(), (name, 'missing_or_symlink')
    assert oct(p.stat().st_mode)[-3:] == '444', (name, 'mode', oct(p.stat().st_mode))
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    assert got == sha, (name, 'sha_mismatch', got, sha)
man = json.loads(Path(pins['freeze'][0]).read_text())
assert str(man.get('r2_status', '')).startswith('FROZEN_EXCLUDED'), man.get('r2_status')
assert man.get('head_sha') == P1_FREEZE_HEAD, ('freeze_head_sha_mismatch', man.get('head_sha'), P1_FREEZE_HEAD)
assert man.get('root') == str(R3)
assert W6_SRC.is_file() and not W6_SRC.is_symlink()
got = hashlib.sha256(W6_SRC.read_bytes()).hexdigest()
assert got == W6_SHA, ('w6_src_mismatch', got)
assert Path(STALE).exists()
print('S0b_AUTHORITY_AND_DATA_PREFLIGHT_OK')
print('HEAD', HEAD)
print('W6', W6_SHA)
print('R3_FREEZE', 'b63d9ff44ca12554834f72555a62800837302bfedc51e15adac66fb7653f2305')
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S0b rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S0b rc=$rc" | tee -a "$LOG" >/dev/null; fi
exit $rc
