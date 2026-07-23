"""Freeze the rotor3b attempt-10 A/B audit manifest (gate-2 blocker A repair).

Deterministic: enumerate 12 checkpoints (6 rotor3b + 6 FP control) with shas,
every audit JSON with sha, full per-save per-surface exact/total table both arms,
and the named 90/90 bank decision. No science claim beyond the counts.
"""
import hashlib, json, glob, re, os

REPO = "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"
CKPT_DIR = f"{REPO}/calm/hrm/checkpoints"
AUDIT_DIR = f"{REPO}/artifacts/rotor/runs/attempt10_ab_audits"
SAVES = (250, 500, 750, 1000, 1250, 1500)
# math A0 (exhaustive 1255) has no surface suffix; the rest are per-surface.
SURFACES = ["mathA0", "trace_train", "trace_held", "add50s", "add120",
            "add120k5to8", "idfull", "l0c1", "language"]
# Acquisition target vs retained priors (drives the 90/90 gate semantics).
ACQUIRE_TARGET = "trace_train"
RETAINED = ["mathA0", "add50s", "add120", "add120k5to8", "idfull", "l0c1", "language"]
CLOSE_SIBLING = "l0c1"
HELD_DIAGNOSTIC = "trace_held"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_counts(path):
    d = json.load(open(path))
    res = d.get("results", {})
    agg = d.get("aggregate") or {}
    if agg and "n_exact" in agg:
        return agg["n_exact"], agg["n_total"]
    tot = sum(r.get("n_total", 0) for r in res.values())
    ex = sum(r.get("n_exact", 0) for r in res.values())
    return ex, tot


def audit_path(arm, step, surface):
    if surface == "mathA0":
        return f"{AUDIT_DIR}/{arm}_step{step:05d}_audit.json"
    return f"{AUDIT_DIR}/{arm}_step{step:05d}_{surface}_audit.json"


checkpoints = {}
for arm, pat in (("rotor3b", "hrm_text_158_phase3_L0c2K2add60to89TraceRotor3b_*_final_step0*.pt"),
                 ("fp_control", "hrm_text_158_phase3_L0c2K2add60to89Trace_seed0017_*_final_step0*.pt")):
    for ck in sorted(glob.glob(f"{CKPT_DIR}/{pat}")):
        step = int(re.findall(r"step0*(\d+)", os.path.basename(ck))[-1])
        checkpoints[(arm, step)] = {"path": os.path.relpath(ck, REPO), "sha256": sha(ck)}

table = {}
audit_files = {}
for surface in SURFACES:
    table[surface] = {}
    for step in SAVES:
        row = {}
        for arm in ("rotor3b", "fp_control"):
            p = audit_path(arm, step, surface)
            if not os.path.exists(p):
                row[arm] = None
                continue
            ex, tot = load_counts(p)
            row[arm] = {"exact": ex, "total": tot, "frac": round(ex / tot, 4) if tot else None}
            audit_files[os.path.relpath(p, REPO)] = sha(p)
        if row.get("rotor3b") and row.get("fp_control"):
            row["delta"] = row["rotor3b"]["exact"] - row["fp_control"]["exact"]
        table[surface][step] = row

# Named 90/90 bank decision per arm (earliest all-clear; final has no privilege).
def bank_decision(arm):
    per_save = {}
    for step in SAVES:
        acq = table[ACQUIRE_TARGET][step].get(arm)
        acq_ok = bool(acq and acq["frac"] is not None and acq["frac"] >= 0.90)
        ret_ok = True
        ret_detail = {}
        for s in RETAINED:
            cell = table[s][step].get(arm)
            ok = bool(cell and cell["frac"] is not None and cell["frac"] >= 0.90)
            ret_detail[s] = {"frac": cell["frac"] if cell else None, "ge90": ok}
            ret_ok = ret_ok and ok
        per_save[step] = {"acquire_ge90": acq_ok, "retain_all_ge90": ret_ok,
                          "banks": acq_ok and ret_ok, "retain_detail": ret_detail}
    banked = next((s for s in SAVES if per_save[s]["banks"]), None)
    return {"per_save": per_save, "earliest_banked_save": banked}

manifest = {
    "schema": "rotor3b_attempt10_ab_manifest/v1",
    "run": "rotor3b attempt-10 (leak-fixed detach); L0c2-K2-addition-60to89-trace slice",
    "treatment": "rotor3b BUNDLE (SwiGLU remat + 512-wide dim-3 saved-tensor 3-bit rotor fake-quant) vs FP control (none)",
    "ab_discipline": "90/90 bank semantics; wall-clock excluded (allocator env differs); scored bundle-vs-none",
    "code_currency_preflight": "/home/gabe/claw-code-creditdir/transient_fp_credit/rotor3b_attempt10_audit/box_code_currency_preflight.json (code_currency_pass=true, head 52de311)",
    "audit_runner_scripts": {
        "mathA0": {"path": "artifacts/rotor/runs/box_audit_mathA0_runner.sh",
                   "sha256": "c96cf5534f12539a7319d99f3f2e49dafc2930603644d5a2fe80fc20d042e321"},
        "per_surface": {"path": "artifacts/rotor/runs/box_audit_persurface_runner.sh",
                        "sha256": "493a00eda523c5f6f17b49c78b0e5bd7c572654d5ab617d79c873029da67bb0f"},
        "note": "The two frozen scripts reproduce all 108 outputs deterministically on the box "
                "(GTX 1070). Each iterates the 12 checkpoints and invokes the exact per-invocation "
                "command below.",
    },
    "audit_command_template": (
        "PYTHONPATH=. python -u scripts/probe_hrm_text_158.py --ckpt-path <CKPT> <SURFACE_FLAG> "
        "[--max-gen 160 for trace surfaces] --audit-output-json <OUT> --use-cached-ternary-infer "
        "--use-kv-cache-decode --use-batched-probe-eval --probe-batch-size 32"
    ),
    "surface_flag_expansion": {
        "mathA0": "--exhaustive-finite-supports",
        "trace_train": "--l0c2k2-addition-60to89-trace-train-audit --max-gen 160",
        "trace_held": "--l0c2k2-addition-60to89-trace-held-audit --max-gen 160",
        "add50s": "--l0c2k2-addition-50s-audit",
        "add120": "--l0c2k2-addition-120-audit",
        "add120k5to8": "--l0c2k2-addition-120-k5to8-audit",
        "idfull": "--l0c2k1-identity-full-audit",
        "l0c1": "--l0c1-audit",
        "language": "--language-supports",
    },
    "aggregation_script": {"path": "scripts/build_rotor3b_ab_manifest.py",
                           "note": "deterministic re-run reproduces this manifest from the frozen audit JSONs"},
    "gate_semantics": {
        "acquire_target": ACQUIRE_TARGET, "retained_priors": RETAINED,
        "close_sibling": CLOSE_SIBLING, "held_diagnostic": HELD_DIAGNOSTIC,
    },
    "checkpoints": {f"{a}_step{s}": v for (a, s), v in sorted(checkpoints.items())},
    "audit_files_sha256": audit_files,
    "per_surface_table": table,
    "bank_decision": {"rotor3b": bank_decision("rotor3b"), "fp_control": bank_decision("fp_control")},
    "observed_pc_kl_max": 0.164825,
    "claim_boundary": (
        "BEHAVIORAL A/B ONLY, bounded to observed counts (no preregistered equivalence margin). "
        "Supported: on audited RETAINED surfaces the two arms are near-matched — |count delta|<=2 "
        "at every save — and both arms clear all numeric retained-surface gates at saves 500-1500. "
        "NOT gate-outcome-equivalent at EVERY save: at step250 rotor3b l0c1=108/121=0.8926 (<0.90) "
        "while FP=109/121=0.9008 (>=0.90) — a single-save one-row numeric divergence during early "
        "acquisition. On the ACQUISITION target the arms are NOT acquisition-equivalent: rotor3b "
        "trails FP by 30/120 at step1500 (47 vs 77). Neither arm banks (both <90% acquire; procedure "
        "slice, trace_held~0 both). NOT proven here (HYPOTHESIS only): any gradient-magnitude "
        "mechanism ('gradient-transparent at minima' / 'noise where gradients large') — needs a "
        "direct per-arm gradient comparison. Do NOT restate as 'retention-equivalent' or 'rotor3b==FP'."
    ),
}
out = f"{REPO}/artifacts/rotor/runs/rotor3b_attempt10_ab_manifest.json"
json.dump(manifest, open(out, "w"), indent=2)
print("wrote", out)
print("checkpoints:", len(checkpoints), "audit_files:", len(audit_files))
for arm in ("rotor3b", "fp_control"):
    bd = manifest["bank_decision"][arm]["earliest_banked_save"]
    print(f"  {arm}: earliest_banked_save = {bd}")
