set -euo pipefail
: "${R1L_RUNNER_LOG:?R1L_RUNNER_LOG required}"
: "${R1L_ROOT:?R1L_ROOT required}"
: "${R1L_EV:?R1L_EV required}"
LOG="$R1L_RUNNER_LOG"
mkdir -p "$(dirname "$LOG")"
echo "RUNNER_PHASE_BEGIN S0" | tee -a "$LOG" >/dev/null
set +e
{
set -euo pipefail
: "${R1L_ROOT:?}"
: "${R1L_EV:?}"
: "${R1L_RUNNER_LOG:?}"
: "${R1L_PLAN_JSON:?}"
: "${R1L_EXPECTED_PLAN_SHA256:?}"
: "${R1L_GATE1_FREEZE_MANIFEST_PATH:?}"
: "${R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256:?}"
: "${R1L_EXPECTED_CONTENT_DIGEST:?}"

# 0) test-only inject seams must be absent in production (before any mkdir)
if [ -n "${R1L_S5_INJECT_FAIL+x}" ] && [ -n "${R1L_S5_INJECT_FAIL}" ]; then
  echo "S0_INJECT_FAIL_SEAM_SET value=${R1L_S5_INJECT_FAIL}" >&2
  exit 97
fi
if [ -n "${R1L_S5_ALLOW_INJECT_BATTERY+x}" ] && [ -n "${R1L_S5_ALLOW_INJECT_BATTERY}" ]; then
  echo "S0_ALLOW_INJECT_BATTERY_SEAM_SET value=${R1L_S5_ALLOW_INJECT_BATTERY}" >&2
  exit 97
fi
echo S0_INJECT_SEAMS_ABSENT_OK

# 1) absences BEFORE mkdir
test ! -e "$R1L_ROOT"
test ! -e "$R1L_EV"

# 2) plan sha
test -f "$R1L_PLAN_JSON"
PLAN_SHA=$(python3 -c "import hashlib,os; print(hashlib.sha256(open(os.environ['R1L_PLAN_JSON'],'rb').read()).hexdigest())")
test "$PLAN_SHA" = "$R1L_EXPECTED_PLAN_SHA256"

# 3) Gate-artifact RESOLUTION (not format check)
python3 - <<'PY'
import hashlib, json, os, re
from pathlib import Path

def content_digest_from_members(members: dict) -> str:
    parts = []
    for name in sorted(members.keys()):
        sha = members[name]["sha256"]
        parts.append(name.encode("utf-8") + b"\0" + bytes.fromhex(sha))
    return hashlib.sha256(b"".join(parts)).hexdigest()

plan_sha = os.environ["R1L_EXPECTED_PLAN_SHA256"]
exp_fm = os.environ["R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256"]
exp_cd = os.environ["R1L_EXPECTED_CONTENT_DIGEST"]
path = Path(os.environ["R1L_GATE1_FREEZE_MANIFEST_PATH"])
assert path.is_file() and not path.is_symlink(), ("gate_manifest_missing", str(path))
# mode should be 0444 when frozen; dry-run may mint then chmod
raw = path.read_bytes()
got_sha = hashlib.sha256(raw).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}", exp_fm), ("exp_fm_not_64hex", exp_fm)
assert re.fullmatch(r"[0-9a-f]{64}", exp_cd), ("exp_cd_not_64hex", exp_cd)
assert re.fullmatch(r"[0-9a-f]{64}", plan_sha), ("plan_sha_not_64hex", plan_sha)
assert got_sha == exp_fm, ("gate_manifest_sha_mismatch", got_sha, exp_fm)
obj = json.loads(raw.decode("utf-8"))
assert isinstance(obj, dict) and "members" in obj and "CONTENT_DIGEST" in obj
members = obj["members"]
assert isinstance(members, dict) and members
# recompute CONTENT_DIGEST from members table
recomputed = content_digest_from_members(members)
assert recomputed == exp_cd, ("content_digest_recompute_mismatch", recomputed, exp_cd)
assert obj.get("CONTENT_DIGEST") == exp_cd, ("content_digest_field_mismatch", obj.get("CONTENT_DIGEST"), exp_cd)
# members must include the live plan sha just verified
plan_path = Path(os.environ["R1L_PLAN_JSON"])
plan_basename = plan_path.name
# plan-in-members: require the plan basename entry's sha equals live plan sha
assert plan_basename in members, ("plan_basename_missing_from_gate_members", plan_basename, list(members.keys()))
assert members[plan_basename].get("sha256") == plan_sha, (
    "plan_sha_not_bound_to_basename_in_gate_members",
    plan_basename,
    members[plan_basename].get("sha256"),
    plan_sha,
)
found = True
print("S0_GATE_ARTIFACT_RESOLVED_OK")
print("GATE_MANIFEST_SHA", got_sha)
print("CONTENT_DIGEST_RECOMPUTED", recomputed)
print("PLAN_SHA_IN_MEMBERS", True)
PY

# 4) ONLY after absences + resolution: create ROOT skeleton
mkdir -p "$R1L_ROOT"/{artifacts,receipts,logs,code}
mkdir -p "$(dirname "$R1L_RUNNER_LOG")"
: > "$R1L_ROOT/logs/.keep"
echo S0_ABSENT_AND_MKDIR_OK
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "RUNNER_PHASE_END S0 rc=$rc" | tee -a "$LOG" >/dev/null
if [ "$rc" -ne 0 ]; then echo "RUNNER_FAIL S0 rc=$rc" | tee -a "$LOG" >/dev/null; exit $rc; fi
exit 0
