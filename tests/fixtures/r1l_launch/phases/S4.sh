set -euo pipefail
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S4" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
: "${R1L_ROOT:?}"
: "${R1L_LAUNCH_SOURCE_COMMIT_SHA:?R1L_LAUNCH_SOURCE_COMMIT_SHA required (40-hex launch source)}"
python3 - <<'PY'
import hashlib, json, os, stat
from pathlib import Path

ROOT = Path(os.environ['R1L_ROOT'])
# Launch source commit — parameter (not a fixture literal). Distinct from P1-mint freeze head.
import re as _re_launch
_launch = os.environ.get('R1L_LAUNCH_SOURCE_COMMIT_SHA', '').strip()
assert _re_launch.fullmatch(r'[0-9a-f]{40}', _launch), (
    'R1L_LAUNCH_SOURCE_COMMIT_SHA_required_40hex', _launch,
)
HEAD = _launch  # launch_source only

# Guard: freeze must not already exist
fm = ROOT / 'r1l_freeze_manifest.json'
assert not fm.exists(), 'freeze_manifest_already_exists'

# --- CURE 2: chmod FIRST on members (files 0444, nested dirs bottom-up 0555),
# then enumerate post-chmod modes, O_EXCL-write freeze, chmod freeze 0444,
# THEN chmod ROOT 0555 (ROOT must stay writable until freeze exists). ---
all_paths = [p for p in ROOT.rglob('*')]
for p in all_paths:
    if p.is_symlink():
        continue
    if p.is_file():
        os.chmod(p, 0o444)
dirs = [p for p in all_paths if p.is_dir() and not p.is_symlink()]
dirs.sort(key=lambda p: len(p.parts), reverse=True)
for d in dirs:
    os.chmod(d, 0o555)
# NOTE: do NOT chmod ROOT yet — need write to create freeze manifest

# Enumerate COMPLETE entry set with POST-CHMOD modes (nested only; ROOT later)
entries = {}
for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False, followlinks=False):
    dp = Path(dirpath)
    for name in dirnames:
        p = dp / name
        rel = p.relative_to(ROOT).as_posix()
        st = p.lstat()
        entries[rel] = {
            'type': 'dir',
            'mode': stat.S_IMODE(st.st_mode),
        }
    for name in filenames:
        p = dp / name
        rel = p.relative_to(ROOT).as_posix()
        if rel == 'r1l_freeze_manifest.json':
            raise SystemExit('freeze_manifest_appeared_mid_walk')
        st = p.lstat()
        mode = stat.S_IMODE(st.st_mode)
        if p.is_symlink():
            entries[rel] = {
                'type': 'symlink',
                'mode': mode,
                'readlink': os.readlink(p),
            }
        else:
            entries[rel] = {
                'type': 'file',
                'mode': mode,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'size': st.st_size,
            }

# Assert post-chmod modes already correct before writing manifest
mismatches = []
for rel, meta in entries.items():
    if meta['type'] == 'dir' and meta['mode'] != 0o555:
        mismatches.append((rel, 'dir', oct(meta['mode'])))
    if meta['type'] == 'file' and meta['mode'] != 0o444:
        mismatches.append((rel, 'file', oct(meta['mode'])))
assert not mismatches, ('pre_manifest_mode_mismatches', mismatches[:20])

man = {
  'schema': 'r1l_launch_freeze_manifest/v13',
  'root': str(ROOT),
  'head_sha': HEAD,
  'entry_count': len(entries),
  'entries': entries,
  'members_exclude_self': True,
  'self_path': 'r1l_freeze_manifest.json',
  'self_sha_rule': 'printed_to_stdout_and_bound_by_evidence; not recorded inside entries',
  'mode_ledger_policy': (
      'chmod members first (files 0444, nested dirs 0555); enumerate post-chmod modes; '
      'O_EXCL write freeze; chmod freeze 0444; chmod ROOT 0555 last; '
      'fresh_walk compares recorded_mode==current_mode for every entry'
  ),
  'root_mode_after_freeze': '0o555',
  'claim_ceiling': 'R1-L launch/runtime currency only; ready_for_main_science=false',
}
text = json.dumps(man, indent=2, sort_keys=True) + '\n'
fd = os.open(str(fm), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, 'w') as f:
    f.write(text)
self_sha = hashlib.sha256(fm.read_bytes()).hexdigest()
self_size = fm.stat().st_size
os.chmod(fm, 0o444)
assert stat.S_IMODE(fm.lstat().st_mode) == 0o444
# final root lock after freeze is on disk
os.chmod(ROOT, 0o555)
assert stat.S_IMODE(ROOT.lstat().st_mode) == 0o555

# FRESH full-walk: set-equality + recorded mode == current mode for EVERY entry
fresh = {}
for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False, followlinks=False):
    dp = Path(dirpath)
    for name in dirnames:
        p = dp / name
        rel = p.relative_to(ROOT).as_posix()
        fresh[rel] = ('dir', stat.S_IMODE(p.lstat().st_mode))
    for name in filenames:
        p = dp / name
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink():
            fresh[rel] = ('symlink', stat.S_IMODE(p.lstat().st_mode))
        else:
            fresh[rel] = ('file', stat.S_IMODE(p.lstat().st_mode))

recorded = set(entries.keys()) | {'r1l_freeze_manifest.json'}
assert recorded == set(fresh.keys()), (
    'fresh_entry_set_mismatch',
    sorted(recorded - set(fresh.keys()))[:10],
    sorted(set(fresh.keys()) - recorded)[:10],
)
mode_mismatches = []
for rel, meta in entries.items():
    t, mode = fresh[rel]
    if mode != meta['mode']:
        mode_mismatches.append((rel, meta['mode'], mode))
# freeze manifest itself: 0444
assert fresh['r1l_freeze_manifest.json'][1] == 0o444
assert not mode_mismatches, ('mode_ledger_mismatches', len(mode_mismatches), mode_mismatches[:20])
print('S4_MODE_LEDGER_MISMATCHES', len(mode_mismatches), len(entries))

# rehash files
for rel, meta in entries.items():
    if meta['type'] == 'file':
        p = ROOT / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta['sha256'], rel
assert hashlib.sha256(fm.read_bytes()).hexdigest() == self_sha

print('S4_FREEZE_ROOT_OK', len(entries), self_sha, self_size)
print('S4_FREEZE_MANIFEST_SELF_SHA256', self_sha)
print('S4_ENTRY_COUNT', len(entries))
print('S4_MEMBERS_EXCLUDE_SELF', True)
print('S4_COMPLETE_ENTRY_DENOMINATOR', True)
print('S4_CHMOD_FIRST_THEN_ENUMERATE', True)
PY
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S4 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S4 rc=$rc" | tee -a "$LOG" >/dev/null; exit $rc; fi
exit 0
