#!/usr/bin/env bash
# selector_support_consensus_v1 durable launcher — plan v5 @ abb5535
# LOCAL-ONLY: code-currency --skip-fetch; artifact transport without box rsync.
# FORBIDDEN: v0 launcher --sync sites (preflight :131, transport :238).
set -euo pipefail

REPO="${REPO:-/mnt/c/Users/gabes/projects/claw-code-hrm-text-158}"
PACKET="$REPO/artifacts/consensus_prep/selector_support_consensus_v1_chain_relaunch_plan_v1.json"
EXPECTED_PACKET_SHA="${EXPECTED_PACKET_SHA:-685e1262db5307f54a0940c2f3e219e132c7020d547abd23c0987e247763a14c}"
HEAD_EXPECTED="${HEAD_EXPECTED:-abb55357ce412dc30df5364cb488f4ea94ac5c49}"
PARENT_SHA256_EXPECTED="${PARENT_SHA256_EXPECTED:-9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec}"
CREDITDIR="${CREDITDIR:-/home/gabe/claw-code-creditdir/transient_fp_credit}"
GATE_ID="${GATE_ID:-consensus_v1_relaunch_abb5535}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/consensus_v1_launch.log}"
TRIPWIRE_HIT=0
SAMPLER_PID=""
SAMPLER_TEARDOWN_CONFIRMED=false
POST_RUN_RC_TRANSPORT=0
POST_RUN_RC_ANALYZER=0
POST_RUN_RC_CONSUMER=0
POST_RUN_RC_WATCHER=0
FAILED_POST_RUN_STAGE=""

write_terminal_receipt() {
  local parent_after="${1:-$(sha256sum "$PARENT_PT" | awk '{print $1}')}"
  local tripwire_flag="${2:-false}"
  local forced_branch="${3:-}"
  local failed_stage="${4:-$FAILED_POST_RUN_STAGE}"
  PYTHONPATH=. python3 scripts/build_consensus_v1_chain_terminal_receipt.py \
    --chain-root "$CHAIN_ROOT" \
    --parent-before "$PARENT_HASH_BEFORE" \
    --parent-after "$parent_after" \
    --head "$HEAD_NOW" \
    --packet-sha "$EXPECTED_PACKET_SHA" \
    --tripwire-triggered "$tripwire_flag" \
    --forced-outcome-branch "$forced_branch" \
    --failed-post-run-stage "$failed_stage" \
    --transport-rc "$POST_RUN_RC_TRANSPORT" \
    --analyzer-rc "$POST_RUN_RC_ANALYZER" \
    --consumer-rc "$POST_RUN_RC_CONSUMER" \
    --watcher-rc "$POST_RUN_RC_WATCHER"
}

exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "=== V1 LAUNCH START $(date -u +%Y-%m-%dT%H:%M:%SZ) gate=$GATE_ID ==="

cd "$REPO"
ACTUAL_PACKET_SHA=$(sha256sum "$PACKET" | awk '{print $1}')
if [[ "$ACTUAL_PACKET_SHA" != "$EXPECTED_PACKET_SHA" ]]; then
  echo "BLOCKER: packet sha mismatch expected=$EXPECTED_PACKET_SHA actual=$ACTUAL_PACKET_SHA"
  exit 2
fi
HEAD_NOW=$(git rev-parse HEAD)
if [[ "$HEAD_NOW" != "$HEAD_EXPECTED" ]]; then
  echo "BLOCKER: head mismatch expected=$HEAD_EXPECTED actual=$HEAD_NOW"
  exit 2
fi

PARENT_PT=$(python3 -c "import json;print(json.load(open('$PACKET'))['code_pins']['parent_checkpoint'])")
PARENT_HASH_BEFORE=$(sha256sum "$PARENT_PT" | awk '{print $1}')
if [[ "$PARENT_HASH_BEFORE" != "$PARENT_SHA256_EXPECTED" ]]; then
  echo "BLOCKER: parent sha mismatch expected=$PARENT_SHA256_EXPECTED actual=$PARENT_HASH_BEFORE"
  exit 2
fi

CHAIN_ID="selector_support_consensus_v1_$(date -u +%Y%m%dT%H%M%SZ)"
CHAIN_ROOT="$CREDITDIR/$CHAIN_ID"
if [[ -e "$CHAIN_ROOT" ]]; then
  echo "BLOCKER: chain root already exists (no resume): $CHAIN_ROOT"
  exit 2
fi
mkdir -p "$CHAIN_ROOT/prelaunch"
echo "CHAIN_ID=$CHAIN_ID"
echo "CHAIN_ROOT=$CHAIN_ROOT"
echo "PARENT_HASH_BEFORE=$PARENT_HASH_BEFORE"

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

contract = packet["probe_argv_contract"]
shared = contract["shared_flags"]
on_flags = contract["on_only_flags"]
parent = None
parent_sha = None
for idx, tok in enumerate(shared):
    if tok == "--parent":
        parent = shared[idx + 1]
    if tok == "--parent-sha256":
        parent_sha = shared[idx + 1]

arms = [("S44_ord17", 17), ("S44_ord43", 43), ("S44_ord44", 44)]
entries = []
for label, seed in arms:
    for arm in ("on", "off"):
        scratch = str(root / label / arm)
        Path(scratch).mkdir(parents=True, exist_ok=True)
        argv = [
            "python3",
            "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
            *shared,
            "--support-order-seed",
            str(seed),
            "--scratch-root",
            scratch,
        ]
        if arm == "on":
            argv.extend(on_flags)
        entries.append(
            {
                "label": label,
                "arm": arm,
                "support_order_seed": seed,
                "scratch_root": scratch,
                "repo_head_sha": head,
                "parent_sha256": parent_sha,
                "parent": parent,
                "argv": argv,
            }
        )

(pre / "argv_echo.json").write_text(
    json.dumps(
        {
            "schema": "hrm_text_158_consensus_prelaunch_argv_echo/v1",
            "chain_id": chain_id,
            "packet_sha256": packet_sha,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "n_entries": len(entries),
            "entries": entries,
        },
        indent=2,
    )
    + "\n",
)

pinned = packet["pinned_files_manifest"].copy()
(pre / "pinned_files_manifest.json").write_text(json.dumps(pinned, indent=2) + "\n")

(pre / "launch_env_export.json").write_text(
    json.dumps(
        {
            "schema": "hrm_text_158_launch_env_export/v0",
            "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "required_launch_env": packet["launch_env"],
        },
        indent=2,
    )
    + "\n",
)
print("prelaunch metadata written")
PY

SRC_PLUMB="/home/gabe/claw-code-creditdir/transient_fp_credit/selector_support_consensus_v0_20260613T204529Z/prelaunch/v1_plumbing_smoke.json"
if [[ -f "$SRC_PLUMB" ]]; then
  cp "$SRC_PLUMB" "$CHAIN_ROOT/prelaunch/v1_plumbing_smoke.json"
  PYTHONPATH=. python3 scripts/verify_pressure_shape_summary_preflight.py \
    --receipt "$CHAIN_ROOT/prelaunch/v1_plumbing_smoke.json" \
    --out "$CHAIN_ROOT/prelaunch/pressure_shape_summary_preflight.json"
  test "$(python3 -c "import json;print(json.load(open('$CHAIN_ROOT/prelaunch/pressure_shape_summary_preflight.json'))['pass'])")" = "True"
fi

PYTHONPATH=. python3 scripts/hrm_text_158_full_sub2_runtime_readiness.py \
  --fixture pre_full_stack_diagnostic \
  --json-out "$CHAIN_ROOT/prelaunch/sub2_readiness_receipt.json"
python3 -c "import json;d=json.load(open('$CHAIN_ROOT/prelaunch/sub2_readiness_receipt.json')); assert d['ready_for_pre_full_stack_diagnostic'] and not d['ready_for_main_science']"

PYTHONPATH=. python3 scripts/box_lane_code_currency_preflight.py \
  --skip-fetch \
  --chain-id "$CHAIN_ID" \
  --head-expected "$HEAD_EXPECTED" \
  --pinned-manifest "$CHAIN_ROOT/prelaunch/pinned_files_manifest.json" \
  --output "$CHAIN_ROOT/prelaunch/box_code_currency_preflight.json"
python3 -c "import json;d=json.load(open('$CHAIN_ROOT/prelaunch/box_code_currency_preflight.json')); assert d['code_currency_pass']"

echo "PRELAUNCH ALL PASS"

PROBE_RESULTS="$CHAIN_ROOT/probe_results.jsonl"
ARM_METRICS="$CHAIN_ROOT/per_arm_metrics.jsonl"
: > "$PROBE_RESULTS"
: > "$ARM_METRICS"
PROBE_NUM=0

start_host_mem_sampler() {
  local out="$1"
  (
    while true; do
      awk '/MemAvailable/ {print "{\"ts\":\"" strftime("%Y-%m-%dT%H:%M:%SZ", systime(), 1) "\",\"mem_available_kib\":" $2 "}"}' /proc/meminfo >> "$out"
      sleep 5
    done
  ) &
  SAMPLER_PID=$!
}

stop_host_mem_sampler() {
  local pid="${SAMPLER_PID:-}"
  if [[ -z "$pid" ]]; then
    SAMPLER_TEARDOWN_CONFIRMED=false
    return
  fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if kill -0 "$pid" 2>/dev/null; then
    SAMPLER_TEARDOWN_CONFIRMED=false
  else
    SAMPLER_TEARDOWN_CONFIRMED=true
  fi
  SAMPLER_PID=""
}

classify_tripwire() {
  local scratch="$1"
  local result
  result=$(PYTHONPATH=. python3 - <<'PY' "$scratch"
import json, sys
from pathlib import Path
from scripts.build_consensus_v1_chain_terminal_receipt import _classify_tripwire_arm

scratch = Path(sys.argv[1])
branch, _, _, _ = _classify_tripwire_arm(scratch)
if branch:
    print(branch)
PY
)
  if [[ -n "$result" ]]; then
    TRIPWIRE_HIT=1
    return 0
  fi
  if grep -q 'LIVENESS_FAILURE' "$scratch/run.log" 2>/dev/null && grep -q 'checkpoint_payload' "$scratch/run.log" 2>/dev/null; then
    TRIPWIRE_HIT=1
    return 0
  fi
  return 1
}

run_probe() {
  local label="$1" arm="$2"
  local scratch="$CHAIN_ROOT/$label/$arm"
  mkdir -p "$scratch"
  PROBE_NUM=$((PROBE_NUM + 1))
  echo "--- PROBE $PROBE_NUM/6: $label/$arm ---"
  local start_ts end_ts rc wall heartbeats
  start_ts=$(date +%s)

  local host_mem_out="$scratch/host_mem_samples.jsonl"
  : > "$host_mem_out"
  start_host_mem_sampler "$host_mem_out"
  local sampler_pid_at_start="$SAMPLER_PID"

  set +e
  python3 - <<'PY' "$CHAIN_ROOT" "$label" "$arm" "$REPO"
import json, os, subprocess, sys
from pathlib import Path

chain_root, label, arm, repo = sys.argv[1:5]
entries = json.loads((Path(chain_root) / "prelaunch" / "argv_echo.json").read_text())["entries"]
entry = next(e for e in entries if e["label"] == label and e["arm"] == arm)
argv = entry["argv"]
scratch = Path(entry["scratch_root"])
env = os.environ.copy()
env["PYTHONPATH"] = "."
outer = [
    "timeout",
    "900s",
    "bin/watch-wrap",
    "--heartbeat",
    "180",
    "--coalesce",
    "2.0",
    "--progress",
    '"event":\\s*"(start|end|heartbeat)"|checkpoint_tensor_export',
    "--error",
    "Traceback|Error|OOM|Killed|assert|LIVENESS_FAILURE",
    "--",
    *argv,
]
proc = subprocess.run(
    outer,
    cwd=repo,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
(scratch / "run.log").write_text(proc.stdout)
sys.exit(proc.returncode)
PY
  rc=$?
  set -e

  stop_host_mem_sampler
  local sampler_teardown="$SAMPLER_TEARDOWN_CONFIRMED"

  end_ts=$(date +%s)
  wall=$((end_ts - start_ts))
  heartbeats=$(grep -c '"event": "heartbeat"' "$scratch/run.log" 2>/dev/null || true)
  heartbeats=${heartbeats:-0}

  PYTHONPATH=. python3 - <<'PY' "$scratch" "$ARM_METRICS" "$PROBE_NUM" "$label" "$arm" "$sampler_pid_at_start" "$sampler_teardown"
import json, sys
from pathlib import Path

scratch = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
probe_num = int(sys.argv[3])
label = sys.argv[4]
arm = sys.argv[5]
sampler_pid = int(sys.argv[6])
sampler_teardown = sys.argv[7].lower() == "true"

dedupe_key = ("phase", "event", "tensor_index", "tensor_key", "elapsed_since_start_seconds")
seen = set()
starts = 0
dones = 0
rss_curve = []
rss_peak = 0
rss_peak_key = None

def ingest_event(ev: dict) -> None:
    global starts, dones, rss_peak, rss_peak_key
    key = tuple(ev.get(k) for k in dedupe_key)
    if key in seen:
        return
    seen.add(key)
    phase = ev.get("phase")
    event = ev.get("event")
    if phase == "checkpoint_payload" and event == "checkpoint_tensor_export_start":
        starts += 1
    if phase == "checkpoint_payload" and event == "checkpoint_tensor_export_done":
        dones += 1
    if phase == "checkpoint_payload" and event == "rss_sample":
        point = {
            "elapsed_since_start_seconds": ev.get("elapsed_since_start_seconds"),
            "rss_bytes": ev.get("rss_bytes"),
            "rss_peak_bytes": ev.get("rss_peak_bytes"),
            "rss_peak_at_tensor_key": ev.get("rss_peak_at_tensor_key"),
            "tensor_key": ev.get("tensor_key"),
        }
        rss_curve.append(point)
        peak = int(ev.get("rss_peak_bytes") or ev.get("rss_bytes") or 0)
        if peak >= rss_peak:
            rss_peak = peak
            rss_peak_key = ev.get("rss_peak_at_tensor_key") or ev.get("tensor_key")

for line in (scratch / "run.log").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(payload, dict):
        ingest_event(payload)

receipt_path = scratch / "receipt.json"
if receipt_path.is_file():
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for ev in receipt.get("phase_telemetry", {}).get("events", []):
            if isinstance(ev, dict):
                ingest_event(ev)
    except (json.JSONDecodeError, OSError):
        pass

(scratch / "b2_rss_curve.json").write_text(json.dumps(rss_curve, indent=2) + "\n", encoding="utf-8")

host_min = None
host_path = scratch / "host_mem_samples.jsonl"
if host_path.is_file():
    for line in host_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            val = int(row.get("mem_available_kib"))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        host_min = val if host_min is None else min(host_min, val)

row = {
    "probe_num": probe_num,
    "label": label,
    "arm": arm,
    "scratch_root": str(scratch),
    "tensor_export_start_count_unique": starts,
    "tensor_export_done_count_unique": dones,
    "export_rss_peak_bytes": rss_peak,
    "export_rss_peak_gib": round(rss_peak / (1024**3), 4) if rss_peak else 0.0,
    "rss_peak_at_tensor_key": rss_peak_key,
    "rss_curve_path": str(scratch / "b2_rss_curve.json"),
    "host_mem_sampler_pid": sampler_pid,
    "host_mem_sampler_teardown_confirmed": sampler_teardown,
    "host_mem_min_available_kib_during_checkpoint_payload": host_min,
    "host_mem_min_available_gib_during_checkpoint_payload": round(host_min / (1024**2), 4) if host_min is not None else None,
}
with metrics_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(row, sort_keys=True))
PY

  if classify_tripwire "$scratch"; then
    echo "TRIPWIRE: checkpoint_payload failure on $label/$arm — STOP chain"
  fi

  local append_rc=0
  set +e
  PYTHONPATH=. python3 - <<'PY' "$PROBE_RESULTS" "$PROBE_NUM" "$label" "$arm" "$rc" "$wall" "$heartbeats" "$scratch"
import sys
from pathlib import Path
from calm.hrm_text_158.native_full_stack.consensus_probe_result_writer import append_probe_result_jsonl

append_probe_result_jsonl(
    Path(sys.argv[1]),
    probe_num=sys.argv[2],
    label=sys.argv[3],
    arm=sys.argv[4],
    exit_code=sys.argv[5],
    wall_s=sys.argv[6],
    heartbeats=sys.argv[7],
    scratch_root=Path(sys.argv[8]),
)
PY
  append_rc=$?
  set -e
  if [[ $append_rc -ne 0 ]]; then
    echo "LAUNCHER_AGGREGATION_FAIL: append_probe_result_jsonl rc=$append_rc probe=$PROBE_NUM label=$label arm=$arm"
    local parent_at_stop
    parent_at_stop=$(sha256sum "$PARENT_PT" | awk '{print $1}')
    write_terminal_receipt "$parent_at_stop" "false" "v_launcher_aggregation_fail" "launcher_append"
    exit 1
  fi

  local receipt_exists=false
  if [[ -f "$scratch/receipt.json" ]]; then
    receipt_exists=true
  fi
  if [[ $rc -ne 0 ]] || [[ "$receipt_exists" != "true" ]] || [[ $TRIPWIRE_HIT -eq 1 ]]; then
    echo "PROBE FAIL: $label/$arm rc=$rc receipt=$receipt_exists tripwire=$TRIPWIRE_HIT"
    local parent_at_stop
    parent_at_stop=$(sha256sum "$PARENT_PT" | awk '{print $1}')
    local forced_branch="v_partial_chain_fail"
    if [[ $TRIPWIRE_HIT -eq 1 ]]; then
      forced_branch=$(PYTHONPATH=. python3 - <<'PY' "$scratch"
import sys
from pathlib import Path
from scripts.build_consensus_v1_chain_terminal_receipt import _classify_tripwire_arm
branch, _, _, _ = _classify_tripwire_arm(Path(sys.argv[1]))
print(branch or "v_partial_chain_fail")
PY
)
    fi
    write_terminal_receipt "$parent_at_stop" "true" "$forced_branch" ""
    exit 1
  fi
}

for spec in "S44_ord17 on" "S44_ord17 off" "S44_ord43 on" "S44_ord43 off" "S44_ord44 on" "S44_ord44 off"; do
  set -- $spec
  run_probe "$1" "$2"
done

PARENT_HASH_AFTER=$(sha256sum "$PARENT_PT" | awk '{print $1}')
if [[ "$PARENT_HASH_AFTER" != "$PARENT_HASH_BEFORE" ]]; then
  echo "BLOCKER: parent hash drift before=$PARENT_HASH_BEFORE after=$PARENT_HASH_AFTER"
  write_terminal_receipt "$PARENT_HASH_AFTER" "false" "" "parent_drift"
  exit 2
fi

echo "ALL 6 PROBES PASS — post-run chain (local-only transport)"

set +e
PYTHONPATH=. python3 scripts/box_lane_artifact_transport.py \
  --chain-id "$CHAIN_ID" \
  --chain-log "$CHAIN_ROOT/producer_science_chain.log" \
  --primary-label S44_ord44 \
  --isolation-label S44_ord43 \
  --corroboration-label S44_ord17
POST_RUN_RC_TRANSPORT=$?

PYTHONPATH=. python3 scripts/analyze_selector_support_invariance.py "$CHAIN_ROOT" --consensus \
  --primary-label S44_ord44 --isolation-label S44_ord43 --corroboration-label S44_ord17 \
  --repo-head "$HEAD_EXPECTED"
POST_RUN_RC_ANALYZER=$?

PYTHONPATH=. python3 scripts/box_lane_consensus_consumer_audit.py \
  --chain-root "$CHAIN_ROOT" --consensus \
  --primary-label S44_ord44 --isolation-label S44_ord43 --corroboration-label S44_ord17 \
  --transport-manifest "$CHAIN_ROOT/box_artifact_transport.json" \
  --chain-log "$CHAIN_ROOT/producer_science_chain.log"
POST_RUN_RC_CONSUMER=$?

PYTHONPATH=. python3 scripts/box_lane_chain_watcher.py \
  "$CHAIN_ROOT/producer_science_chain.log" \
  --manifest "$CHAIN_ROOT/box_lane_overlap_manifest.json" \
  --waive
POST_RUN_RC_WATCHER=$?
set -e

if [[ $POST_RUN_RC_TRANSPORT -ne 0 ]]; then FAILED_POST_RUN_STAGE="transport"; fi
if [[ $POST_RUN_RC_ANALYZER -ne 0 && -z "$FAILED_POST_RUN_STAGE" ]]; then FAILED_POST_RUN_STAGE="analyzer"; fi
if [[ $POST_RUN_RC_CONSUMER -ne 0 && -z "$FAILED_POST_RUN_STAGE" ]]; then FAILED_POST_RUN_STAGE="consumer"; fi

write_terminal_receipt "$PARENT_HASH_AFTER" "false" "" "$FAILED_POST_RUN_STAGE"

echo "=== V1 LAUNCH COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) chain=$CHAIN_ID ==="
