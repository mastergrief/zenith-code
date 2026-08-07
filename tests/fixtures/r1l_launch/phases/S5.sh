set -euo pipefail
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S5" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
# S5 entry: re-assert test-only inject seam absent (production path)
# Dry battery may set R1L_S5_ALLOW_INJECT_BATTERY=1 to exercise inject points.
if [ -n "${R1L_S5_INJECT_FAIL+x}" ] && [ -n "${R1L_S5_INJECT_FAIL}" ]; then
  if [ "${R1L_S5_ALLOW_INJECT_BATTERY:-}" != "1" ]; then
    echo "S5_INJECT_FAIL_SEAM_SET_AT_ENTRY value=${R1L_S5_INJECT_FAIL}" >&2
    exit 97
  fi
  echo "S5_INJECT_BATTERY_OVERRIDE_ACTIVE value=${R1L_S5_INJECT_FAIL}"
fi
: "${R1L_ROOT:?}"
: "${R1L_EV:?}"
: "${R1L_RUNNER_LOG:?}"
: "${R1L_PLAN_JSON:?}"
: "${R1L_EXPECTED_PLAN_SHA256:?}"
: "${R1L_GATE1_FREEZE_MANIFEST_PATH:?}"
: "${R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256:?}"
: "${R1L_EXPECTED_CONTENT_DIGEST:?}"
python3 - <<'PY'
import hashlib, json, os, re, stat
from pathlib import Path

ROOT = Path(os.environ['R1L_ROOT'])
EV = Path(os.environ['R1L_EV'])
RUNNER_LOG = Path(os.environ['R1L_RUNNER_LOG'])
PLAN_JSON = Path(os.environ['R1L_PLAN_JSON'])
HEAD = '0636177fbe52d8c6ff5db71312f51240b31fceb2'
EXP_PLAN = os.environ['R1L_EXPECTED_PLAN_SHA256']
EXP_FM = os.environ['R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256']
EXP_CD = os.environ['R1L_EXPECTED_CONTENT_DIGEST']
GATE_PATH = Path(os.environ['R1L_GATE1_FREEZE_MANIFEST_PATH'])

def content_digest_from_members(members: dict) -> str:
    parts = []
    for name in sorted(members.keys()):
        sha = members[name]["sha256"]
        parts.append(name.encode("utf-8") + b"\0" + bytes.fromhex(sha))
    return hashlib.sha256(b"".join(parts)).hexdigest()

assert RUNNER_LOG.is_file(), 'runner_log_missing'
log_text = RUNNER_LOG.read_text(errors='replace')
n_pass = len(re.findall(r'(?m)^R1L_TERMINAL_PASS\s*$', log_text))
n_fail = len(re.findall(r'(?m)^R1L_TERMINAL_FAIL\s*$', log_text))
n_rfail = len(re.findall(r'(?m)^RUNNER_FAIL\b', log_text))
assert n_pass >= 1, ('expected_R1L_TERMINAL_PASS_present', n_pass)
assert n_fail == 0, ('unexpected_R1L_TERMINAL_FAIL', n_fail)
assert n_rfail == 0, ('unexpected_RUNNER_FAIL', n_rfail)
terminal = 'R1L_TERMINAL_PASS'
print('S5_TERMINAL_VERIFIED_FROM_LOG', terminal, 'n_pass', n_pass)

raw = GATE_PATH.read_bytes()
got_sha = hashlib.sha256(raw).hexdigest()
assert got_sha == EXP_FM, ('s5_gate_manifest_sha_mismatch', got_sha, EXP_FM)
gobj = json.loads(raw.decode())
recomputed = content_digest_from_members(gobj['members'])
assert recomputed == EXP_CD, ('s5_content_digest_recompute_mismatch', recomputed, EXP_CD)
assert gobj.get('CONTENT_DIGEST') == EXP_CD
plan_sha = hashlib.sha256(PLAN_JSON.read_bytes()).hexdigest()
assert plan_sha == EXP_PLAN
plan_basename = PLAN_JSON.name
assert plan_basename in gobj['members'], ('plan_basename_missing_from_gate_members', plan_basename)
assert gobj['members'][plan_basename].get('sha256') == plan_sha
print('S5_GATE_ARTIFACT_RERESOLVED_OK')

assert not EV.exists(), 'evidence_root_exists'
os.mkdir(EV)

fm = ROOT/'r1l_freeze_manifest.json'
freeze_bytes = fm.read_bytes()
freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
man = json.loads(freeze_bytes)
assert man.get('members_exclude_self') is True

fresh = {}
for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False, followlinks=False):
    dp = Path(dirpath)
    for name in dirnames:
        p = dp / name
        fresh[p.relative_to(ROOT).as_posix()] = ('dir', stat.S_IMODE(p.lstat().st_mode))
    for name in filenames:
        p = dp / name
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink():
            fresh[rel] = ('symlink', stat.S_IMODE(p.lstat().st_mode))
        else:
            fresh[rel] = ('file', stat.S_IMODE(p.lstat().st_mode))
recorded = set(man['entries'].keys()) | {'r1l_freeze_manifest.json'}
assert recorded == set(fresh.keys()), ('s5_fresh_mismatch', sorted(recorded ^ set(fresh.keys()))[:10])
mode_mm = []
for rel, meta in man['entries'].items():
    t, mode = fresh[rel]
    if mode != meta['mode']:
        mode_mm.append((rel, meta['mode'], mode))
    if meta['type'] == 'file':
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == meta['sha256'], rel
assert not mode_mm, ('s5_mode_ledger_mismatches', len(mode_mm), mode_mm[:10])
print('S5_MODE_LEDGER_MISMATCHES', len(mode_mm), len(man['entries']))

def oexcl_write(path: Path, text: str):
    data = text.encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)

exec_log = {
  'schema': 'r1l_v13_execution_log',
  'root': str(ROOT),
  'phases': ['S0','S0b','S1','S2','S3','S4','S5'],
  'terminal': terminal,
  'terminal_source': 'verified_from_R1L_RUNNER_LOG',
  'r1l_freeze_manifest_sha256': freeze_sha,
  'reviewed_plan_json_sha256': plan_sha,
  'gate1_freeze_manifest_sha256': EXP_FM,
  'CONTENT_DIGEST': EXP_CD,
  'external_runner_log': str(RUNNER_LOG),
  'stop_marker_protocol': {
    'observer_terminal_authority': 'watch-wrap spawn [EXIT rc=]',
    'stop_on': 'absent under topology (c)',
    'runner_log_markers': 'auxiliary record; not a stop channel',
    'R1L_TERMINAL_PASS_role': 'progress_informational_only',
    'order': [
      'S5 tee body: re-assert inject seam absent; verify log + write execution_log/evidence_manifest only',
      'unteed: project sha256(log||RUNNER_PASS\\n); write attestation; exact EV denom; chmod; mode re-assert',
      'unteed LAST: echo RUNNER_PASS >> log; arc=$?; exit $arc (append status survives)',
      'observer: watch-wrap spawn [EXIT rc=] is primary terminal; --stop-on empty under topology (c)',
      'terminal receipt: [EXIT rc=0] AND attestation present AND actual==projected AND last non-empty line RUNNER_PASS count==1',
    ],
  },
}
log_path = EV/'execution_log.json'
man_path = EV/'evidence_manifest.json'
oexcl_write(log_path, json.dumps(exec_log, indent=2, sort_keys=True)+'\n')
execution_log_sha256 = hashlib.sha256(log_path.read_bytes()).hexdigest()

artifact_sha256 = {}
for rel in [
    'launch_manifest.json', 'launch_env.json', 'authority_binding.json',
    'receipts/r1l_launch_runtime_receipt.json', 'receipts/launch_log_at_mint.log',
    'r1l_freeze_manifest.json', 'logs/launch.log',
]:
    p = ROOT/rel
    assert p.is_file(), rel
    artifact_sha256[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

payload = {
  'schema': 'r1l_v13_execution_evidence_manifest',
  'r1l_root': str(ROOT),
  'r1l_freeze_manifest_sha256': freeze_sha,
  'r1l_freeze_manifest_size': len(freeze_bytes),
  'head_sha': HEAD,
  'plan_json_path': str(PLAN_JSON),
  'reviewed_plan_json_sha256': plan_sha,
  'gate1_freeze_manifest_path': str(GATE_PATH),
  'gate1_freeze_manifest_sha256': EXP_FM,
  'CONTENT_DIGEST': EXP_CD,
  'execution_log_sha256': execution_log_sha256,
  'execution_log_path': 'execution_log.json',
  'terminal_log_attestation_path': 'terminal_log_attestation.json',
  'terminal_log_attestation_note': (
      'written in unteed finalize BEFORE RUNNER_PASS append; binds projected_external_runner_log_sha256; '
      'reviewer must recompute actual log digest post-run'
  ),
  'terminal_log_attestation_required_fields': [
    'schema', 'r1l_root', 'reviewed_plan_json_sha256', 'gate1_freeze_manifest_sha256',
    'CONTENT_DIGEST', 'execution_log_sha256', 'evidence_manifest_sha256',
    'external_runner_log_path', 'projected_external_runner_log_sha256',
    'final_marker_position', 'final_marker_name', 'hash_after_last_write_protocol',
    'reviewer_obligation',
  ],
  'terminal_log_attestation_closure_check': (
      'from EV alone: required fields present; attestation.evidence_manifest_sha256 == sha256(evidence_manifest); '
      'execution_log/plan/gate/CD/root match; terminal receipt: actual runner log sha == projected and last line RUNNER_PASS'
  ),
  'artifact_sha256': artifact_sha256,
  'member_names_exact_post_finalize': [
    'evidence_manifest.json', 'execution_log.json', 'terminal_log_attestation.json',
  ],
}
oexcl_write(man_path, json.dumps(payload, indent=2, sort_keys=True)+'\n')
print('S5_EVIDENCE_ROOT_CORE_OK', hashlib.sha256(man_path.read_bytes()).hexdigest())
print('S5_FREEZE_MANIFEST_SELF_SHA256', freeze_sha)
print('S5_EXECUTION_LOG_SHA256', execution_log_sha256)
print('S5_REVIEWED_PLAN_JSON_SHA256', plan_sha)
print('S5_OEXCL', True)
print('S5_AWAITING_PROJECTED_FINALIZE', True)
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S5 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S5 rc=$rc" | tee -a "$LOG" >/dev/null; exit $rc; fi
# --- S5 projected-digest finalize (UNTEED); append status is returned ---
set +e
python3 - <<'PY'
import hashlib, json, os, stat, time
from pathlib import Path

def die(msg):
    raise SystemExit(msg)

inject = os.environ.get('R1L_S5_INJECT_FAIL', '').strip()
log = Path(os.environ['R1L_RUNNER_LOG'])
ev = Path(os.environ['R1L_EV'])
root = Path(os.environ['R1L_ROOT'])
plan_json = Path(os.environ['R1L_PLAN_JSON'])
gate_path = Path(os.environ['R1L_GATE1_FREEZE_MANIFEST_PATH'])
MARKER = b'RUNNER_PASS\n'

cur1 = log.read_bytes()
time.sleep(0.1)
cur2 = log.read_bytes()
assert cur1 == cur2, ('runner_log_unstable_before_project', hashlib.sha256(cur1).hexdigest(), hashlib.sha256(cur2).hexdigest())
projected = hashlib.sha256(cur1 + MARKER).hexdigest()
print('S5_PROJECTED_RUNNER_LOG_SHA256', projected)

man_path = ev / 'evidence_manifest.json'
elog_path = ev / 'execution_log.json'
att_path = ev / 'terminal_log_attestation.json'
assert man_path.is_file() and elog_path.is_file()
if inject == 'attestation_exists':
    att_path.write_text('preexisting\n')
assert not att_path.exists(), 'attestation_preexists'

man_bytes = man_path.read_bytes()
elog_bytes = elog_path.read_bytes()
man_obj = json.loads(man_bytes.decode('utf-8'))
ev_manifest_sha = hashlib.sha256(man_bytes).hexdigest()
exec_log_sha = hashlib.sha256(elog_bytes).hexdigest()
assert man_obj.get('execution_log_sha256') == exec_log_sha
plan_sha = hashlib.sha256(plan_json.read_bytes()).hexdigest()
assert plan_sha == os.environ['R1L_EXPECTED_PLAN_SHA256']
assert man_obj.get('reviewed_plan_json_sha256') == plan_sha
assert man_obj.get('r1l_root') == str(root)
gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
assert gate_sha == os.environ['R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256']
assert man_obj.get('gate1_freeze_manifest_sha256') == gate_sha
cd = os.environ['R1L_EXPECTED_CONTENT_DIGEST']
assert man_obj.get('CONTENT_DIGEST') == cd
req = list(man_obj.get('terminal_log_attestation_required_fields') or [])
assert req, 'evidence_manifest_missing_attestation_required_fields'

if inject == 'closure':
    plan_sha = '0' * 64

att = {
  'schema': 'r1l_v13_terminal_log_attestation',
  'r1l_root': str(root),
  'reviewed_plan_json_sha256': plan_sha,
  'gate1_freeze_manifest_sha256': gate_sha,
  'CONTENT_DIGEST': cd,
  'execution_log_sha256': exec_log_sha,
  'evidence_manifest_sha256': ev_manifest_sha,
  'external_runner_log_path': str(log),
  'projected_external_runner_log_sha256': projected,
  'final_marker_position': 'appended_last_by_finalize',
  'final_marker_name': 'RUNNER_PASS',
  'hash_after_last_write_protocol': 'projected_digest_then_append_marker',
  'reviewer_obligation': (
      'terminal receipt: attestation present AND '
      'sha256(runner_log_bytes) == projected_external_runner_log_sha256 AND '
      'last non-empty line == RUNNER_PASS with count == 1'
  ),
}
missing = [k for k in req if k not in att]
assert not missing, ('attestation_missing_required_fields', missing)
if inject == 'attestation_write':
    att_path.mkdir()
data = (json.dumps(att, indent=2, sort_keys=True) + '\n').encode()
fd = os.open(str(att_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, 'wb') as f:
    f.write(data)

att2 = json.loads(att_path.read_text())
assert att2['evidence_manifest_sha256'] == hashlib.sha256(man_path.read_bytes()).hexdigest()
assert att2['execution_log_sha256'] == hashlib.sha256(elog_path.read_bytes()).hexdigest()
assert att2['reviewed_plan_json_sha256'] == man_obj['reviewed_plan_json_sha256']
assert att2['gate1_freeze_manifest_sha256'] == man_obj['gate1_freeze_manifest_sha256']
assert att2['CONTENT_DIGEST'] == man_obj['CONTENT_DIGEST']
assert att2['r1l_root'] == man_obj['r1l_root']
assert att2['projected_external_runner_log_sha256'] == projected
print('S5_ATTESTATION_CLOSURE_OK', True)

expected = {'evidence_manifest.json', 'execution_log.json', 'terminal_log_attestation.json'}
if inject == 'membership_extra':
    (ev / 'unexpected_extra.txt').write_text('x\n')
if inject == 'membership_missing':
    os.chmod(elog_path, 0o644)
    elog_path.unlink()
names = set()
for p in ev.iterdir():
    if p.name in ('.', '..'):
        continue
    names.add(p.name)
assert names == expected, ('ev_membership_mismatch', sorted(names), sorted(expected))
print('S5_EV_MEMBERSHIP_EXACT', sorted(names))

if inject == 'chmod':
    for p in ev.rglob('*'):
        if p.is_file() and p.name != 'terminal_log_attestation.json' and not p.is_symlink():
            os.chmod(p, 0o444)
        elif p.is_dir() and not p.is_symlink():
            os.chmod(p, 0o555)
else:
    for p in list(ev.rglob('*')):
        if p.is_symlink():
            continue
        if p.is_file():
            os.chmod(p, 0o444)
    dirs = [p for p in ev.rglob('*') if p.is_dir() and not p.is_symlink()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        os.chmod(d, 0o555)
    os.chmod(ev, 0o555)

mode_mm = []
fresh = {}
for dirpath, dirnames, filenames in os.walk(ev, topdown=False, followlinks=False):
    dp = Path(dirpath)
    for name in dirnames:
        p = dp / name
        mode = stat.S_IMODE(p.lstat().st_mode)
        fresh[p.relative_to(ev).as_posix()] = ('dir', mode)
        if mode != 0o555:
            mode_mm.append((str(p), oct(mode), 'dir'))
    for name in filenames:
        p = dp / name
        mode = stat.S_IMODE(p.lstat().st_mode)
        fresh[p.relative_to(ev).as_posix()] = ('file', mode)
        if mode != 0o444:
            mode_mm.append((str(p), oct(mode), 'file'))
root_mode = stat.S_IMODE(ev.lstat().st_mode)
if root_mode != 0o555:
    mode_mm.append((str(ev), oct(root_mode), 'ev_root'))
if inject == 'mode_assert':
    mode_mm.append(('injected', '0o666', 'file'))
assert not mode_mm, ('ev_mode_ledger_mismatches', mode_mm)
print('S5_EV_MODE_LEDGER_MISMATCHES', 0, len(fresh))
print('S5_PROJECTED_FINALIZE_READY', projected)
print('S5_NO_RUNNER_PASS_YET', True)
PY
frc=$?
if [ "$frc" -ne 0 ]; then echo "RUNNER_FAIL S5 rc=$frc" | tee -a "$LOG" >/dev/null; exit $frc; fi
# Sole success authority: append status is returned (no unconditional exit 0)
if [ "${R1L_S5_INJECT_FAIL:-}" = "append" ]; then
  chmod a-w "$LOG" 2>/dev/null || true
fi
echo "RUNNER_PASS" >> "$LOG"
arc=$?
if [ "$arc" -ne 0 ]; then
  echo "RUNNER_FAIL S5 append_rc=$arc TERMINAL_MARKER_UNWRITABLE" | tee -a "$LOG" >/dev/null 2>&1 || true
  echo "S5_TERMINAL_MARKER_UNWRITABLE append_rc=$arc" >&2
  exit $arc
fi
exit $arc
