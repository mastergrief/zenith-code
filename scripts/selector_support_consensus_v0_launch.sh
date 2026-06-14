#!/usr/bin/env bash
# selector_support_consensus_v0 durable launcher — packet 2b1ec738 @ ba0ecb05
set -euo pipefail

REPO="${REPO:-/mnt/c/Users/gabes/projects/claw-code-hrm-text-158}"
PACKET="$REPO/artifacts/consensus_prep/selector_support_consensus_v0_launch_packet.json"
EXPECTED_PACKET_SHA="${EXPECTED_PACKET_SHA:-2b1ec7382b3e84636540a410378542451bef1f5773e2221fe0d1c94b1868068c}"
HEAD_EXPECTED="${HEAD_EXPECTED:-ba0ecb051fbdd825689e86efe6fc9ab0d93d8574}"
CREDITDIR="${CREDITDIR:-/home/gabe/claw-code-creditdir/transient_fp_credit}"
GATE_ID="${GATE_ID:-checkpoint_payload_fix_b1}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/consensus_v0_launch.log}"
TRIPWIRE_HIT=0

exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "=== LAUNCH START $(date -u +%Y-%m-%dT%H:%M:%SZ) gate=$GATE_ID ==="

cd "$REPO"
ACTUAL_SHA=$(sha256sum "$PACKET" | awk '{print $1}')
if [[ "$ACTUAL_SHA" != "$EXPECTED_PACKET_SHA" ]]; then
  echo "BLOCKER: packet sha mismatch expected=$EXPECTED_PACKET_SHA actual=$ACTUAL_SHA"
  exit 2
fi
HEAD_NOW=$(git rev-parse HEAD)
if [[ "$HEAD_NOW" != "$HEAD_EXPECTED" ]]; then
  echo "BLOCKER: head mismatch expected=$HEAD_EXPECTED actual=$HEAD_NOW"
  exit 2
fi

CHAIN_ID="selector_support_consensus_v0_$(date -u +%Y%m%dT%H%M%SZ)"
CHAIN_ROOT="$CREDITDIR/$CHAIN_ID"
mkdir -p "$CHAIN_ROOT/prelaunch"
echo "CHAIN_ID=$CHAIN_ID"
echo "CHAIN_ROOT=$CHAIN_ROOT"

export HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1
export HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1

python3 - <<'PY' "$CHAIN_ID" "$CHAIN_ROOT" "$PACKET" "$HEAD_EXPECTED" "$EXPECTED_PACKET_SHA"
import json, sys
from datetime import datetime, timezone
from pathlib import Path

chain_id, chain_root, packet_path, head, packet_sha = sys.argv[1:6]
packet = json.loads(Path(packet_path).read_text())
root = Path(chain_root)
pre = root / "prelaunch"
pre.mkdir(parents=True, exist_ok=True)

shared = packet["prelaunch_argv_echo"]["shared_probe_flags"]
on_flags = packet["prelaunch_argv_echo"]["on_only_flags"]
parent = None
parent_sha = None
for tok in shared:
    if tok == "--parent" and parent is None:
        idx = shared.index(tok)
        parent = shared[idx+1]
    if tok == "--parent-sha256":
        idx = shared.index(tok)
        parent_sha = shared[idx+1]

arms = [
    ("S44_ord17", 17), ("S44_ord43", 43), ("S44_ord44", 44),
]
entries = []
for label, seed in arms:
    for arm in ("on", "off"):
        scratch = str(root / label / arm)
        Path(scratch).mkdir(parents=True, exist_ok=True)
        argv = [
            "python3", "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
            *shared,
            "--support-order-seed", str(seed),
            "--scratch-root", scratch,
        ]
        if arm == "on":
            argv.extend(on_flags)
        entries.append({
            "label": label, "arm": arm, "support_order_seed": seed,
            "scratch_root": scratch, "repo_head_sha": head,
            "parent_sha256": parent_sha, "parent": parent, "argv": argv,
        })

argv_echo = {
    "schema": "hrm_text_158_consensus_prelaunch_argv_echo/v1",
    "chain_id": chain_id,
    "packet_sha256": packet_sha,
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "n_entries": len(entries),
    "entries": entries,
}
(pre / "argv_echo.json").write_text(json.dumps(argv_echo, indent=2) + "\n")

pinned = packet["pinned_files_manifest"].copy()
(pre / "pinned_files_manifest.json").write_text(json.dumps(pinned, indent=2) + "\n")

f1 = packet["f1_budget"]
(pre / "f1_budget_cite.json").write_text(json.dumps({
    "schema": "hrm_text_158_f1_budget_cite/v0",
    "source_path": "/home/gabe/claw-code-creditdir/transient_fp_credit/f1_step_update_discriminator_20260612/step_update_cost_attribution_discriminator_v1.json",
    "threshold_s": f1["threshold_s"],
    "enabled_path_floor_s_per_step": f1["enabled_path_floor_s_per_step"],
    "step_update_headroom_pct": f1["step_update_headroom_pct"],
}, indent=2) + "\n")

(pre / "launch_env_export.json").write_text(json.dumps({
    "schema": "hrm_text_158_launch_env_export/v0",
    "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "required_launch_env": {
        "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
    },
    "empirical_proof": "values read from operator shell immediately before probe loop; subprocess inherits same environment",
}, indent=2) + "\n")
print("prelaunch metadata written")
PY

SRC_PLUMB="/home/gabe/claw-code-creditdir/transient_fp_credit/selector_support_consensus_v0_20260613T204529Z/prelaunch/v1_plumbing_smoke.json"
cp "$SRC_PLUMB" "$CHAIN_ROOT/prelaunch/v1_plumbing_smoke.json"
PYTHONPATH=. python3 scripts/verify_pressure_shape_summary_preflight.py \
  --receipt "$CHAIN_ROOT/prelaunch/v1_plumbing_smoke.json" \
  --out "$CHAIN_ROOT/prelaunch/pressure_shape_summary_preflight.json"
test "$(python3 -c "import json;print(json.load(open('$CHAIN_ROOT/prelaunch/pressure_shape_summary_preflight.json'))['pass'])")" = "True"

PYTHONPATH=. python3 scripts/hrm_text_158_full_sub2_runtime_readiness.py \
  --fixture pre_full_stack_diagnostic \
  --json-out "$CHAIN_ROOT/prelaunch/sub2_readiness_receipt.json"
python3 -c "import json;d=json.load(open('$CHAIN_ROOT/prelaunch/sub2_readiness_receipt.json')); assert d['ready_for_pre_full_stack_diagnostic'] and not d['ready_for_main_science']"

PYTHONPATH=. python3 scripts/box_lane_code_currency_preflight.py \
  --sync --chain-id "$CHAIN_ID" \
  --head-expected "$HEAD_EXPECTED" \
  --pinned-manifest "$CHAIN_ROOT/prelaunch/pinned_files_manifest.json" \
  --output "$CHAIN_ROOT/prelaunch/box_code_currency_preflight.json" \
  --skip-fetch
python3 -c "import json;d=json.load(open('$CHAIN_ROOT/prelaunch/box_code_currency_preflight.json')); assert d['code_currency_pass']; assert all(f['match'] for f in d['files'])"

echo "PRELAUNCH ALL PASS"

PROBE_RESULTS="$CHAIN_ROOT/probe_results.jsonl"
: > "$PROBE_RESULTS"
PROBE_NUM=0

run_probe() {
  local label="$1" arm="$2"
  local scratch="$CHAIN_ROOT/$label/$arm"
  mkdir -p "$scratch"
  PROBE_NUM=$((PROBE_NUM + 1))
  echo "--- PROBE $PROBE_NUM/6: $label/$arm ---"
  local start_ts end_ts rc wall heartbeats
  start_ts=$(date +%s)
  set +e
  python3 - <<'PY' "$CHAIN_ROOT" "$label" "$arm" "$REPO"
import json, os, subprocess, sys
from pathlib import Path
chain_root, label, arm, repo = sys.argv[1:5]
entries = json.loads((Path(chain_root)/"prelaunch"/"argv_echo.json").read_text())["entries"]
entry = next(e for e in entries if e["label"]==label and e["arm"]==arm)
argv = entry["argv"]
scratch = Path(entry["scratch_root"])
env = os.environ.copy()
env["PYTHONPATH"] = "."
proc = subprocess.run(
    ["bin/watch-wrap", "--heartbeat", "180", "--coalesce", "2.0",
     "--progress", '"event":\\s*"(start|end|heartbeat)"',
     "--error", "Traceback|Error|OOM|Killed|assert|LIVENESS_FAILURE",
     "--", *argv],
    cwd=repo,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
(scratch/"run.log").write_text(proc.stdout)
sys.exit(proc.returncode)
PY
  rc=$?
  set -e
  end_ts=$(date +%s)
  wall=$((end_ts - start_ts))
  heartbeats=$(grep -c '"event": "heartbeat"' "$scratch/run.log" 2>/dev/null || echo 0)

  if grep -q 'LIVENESS_FAILURE' "$scratch/run.log" && grep -q 'checkpoint_payload' "$scratch/run.log"; then
    elapsed=$(python3 -c "
import json
from pathlib import Path
p=Path('$scratch/last_active_phase.json')
if p.exists():
 d=json.loads(p.read_text())
 print(d.get('active_phase_elapsed_seconds', -1))
else:
 print(-1)
" 2>/dev/null || echo -1)
    if python3 -c "e=float('$elapsed'); import sys; sys.exit(0 if (e>5 and e<250) else 1)"; then
      echo "TRIPWIRE: checkpoint_payload LIVENESS_FAILURE active_phase_elapsed=$elapsed — STOP"
      TRIPWIRE_HIT=1
    fi
  fi

  PYTHONPATH=. python3 - <<'PY' "$PROBE_RESULTS" "$PROBE_NUM" "$label" "$arm" "$rc" "$wall" "$heartbeats" "$scratch"
import sys
from pathlib import Path
from calm.hrm_text_158.native_full_stack.consensus_probe_result_writer import append_probe_result_jsonl

append_probe_result_jsonl(
    Path(sys.argv[1]),
    probe_num=int(sys.argv[2]),
    label=sys.argv[3],
    arm=sys.argv[4],
    exit_code=int(sys.argv[5]),
    wall_s=int(sys.argv[6]),
    heartbeats=int(sys.argv[7]),
    scratch_root=Path(sys.argv[8]),
)
PY

  receipt_exists=false
  if [[ -f "$scratch/receipt.json" ]]; then
    receipt_exists=true
  fi
  if [[ $rc -ne 0 ]] || [[ "$receipt_exists" != "true" ]]; then
    echo "PROBE FAIL: $label/$arm rc=$rc receipt=$receipt_exists"
    if [[ $TRIPWIRE_HIT -eq 1 ]]; then
      echo "TRIPWIRE engaged — no auto-relaunch"
    fi
    exit 1
  fi
}

for spec in "S44_ord17 on" "S44_ord17 off" "S44_ord43 on" "S44_ord43 off" "S44_ord44 on" "S44_ord44 off"; do
  set -- $spec
  run_probe "$1" "$2"
done

echo "ALL 6 PROBES PASS — post-run chain"

PYTHONPATH=. python3 scripts/box_lane_artifact_transport.py \
  --chain-id "$CHAIN_ID" --sync \
  --primary-label S44_ord44 --isolation-label S44_ord43 --corroboration-label S44_ord17 \
  --chain-log "$CHAIN_ROOT/producer_science_chain.log"

PYTHONPATH=. python3 scripts/analyze_selector_support_invariance.py "$CHAIN_ROOT" --consensus \
  --primary-label S44_ord44 --isolation-label S44_ord43 --corroboration-label S44_ord17 \
  --repo-head "$HEAD_EXPECTED"

PYTHONPATH=. python3 scripts/box_lane_consensus_consumer_audit.py \
  --chain-root "$CHAIN_ROOT" --consensus \
  --primary-label S44_ord44 --isolation-label S44_ord43 --corroboration-label S44_ord17 \
  --transport-manifest "$CHAIN_ROOT/box_artifact_transport.json" \
  --chain-log "$CHAIN_ROOT/producer_science_chain.log"

PYTHONPATH=. python3 scripts/box_lane_chain_watcher.py \
  "$CHAIN_ROOT/producer_science_chain.log" \
  --manifest "$CHAIN_ROOT/box_lane_overlap_manifest.json"

echo "=== LAUNCH COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) chain=$CHAIN_ID ==="
