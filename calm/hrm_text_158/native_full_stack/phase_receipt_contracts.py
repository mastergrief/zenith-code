"""Phase-0/1 receipt contracts + aggregate state machine (PLAN_v9).

Extracted behavior-preservingly from forgetting_mechanism_screen_reducers.
"""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM0,
    ARM1,
    ARM2,
    ARM3,
    EPS,
    FAMILY_F4,
    TIE_TOLERANCE_BPW,
    classify_forgetting_family_screen,
)
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    ACQ_N,
    ACQUISITION_SELECTION_SHA256,
    IDENTITY_SELECTION_SHA256,
)

# PLAN_v9 Phase-0 screen identity + phase geometry (launch_classification.phase_budgets)
PHASE0_SCREEN_ID = "forgetting_mechanism_screen/v1"
PHASE0_ARM_ID = ARM0
PHASE_BATCH = 8
PHASE_TOPK = 1024
PHASE0_STEPS = 150
PHASE0B_STEPS_FALLBACK_ONCE = 600
PHASE0_ALLOWED_STEPS = frozenset({PHASE0_STEPS, PHASE0B_STEPS_FALLBACK_ONCE})
DEFAULT_AUTHORITY_DISPATCH = "1784812148229-f466bc29"
DEFAULT_PLAN_SHA256 = (
    "07a02afff92cef7b2c6cee46a761a1e46b6b3422df911f8b4d4f63d41157e7a5"
)
DEFAULT_PARENT_SHA256 = (
    "2d9b9f6746e66cec9e7e39d65e8171151e836daca99df6b56fb488d8a6f2403b"
)
CENSOR_CLEAR_MAX = 0.50


def arm_metrics_for_classifier(arm_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Extract G0/G0b/G1 + H_final fields from a per-arm screen receipt."""
    m = arm_receipt.get("measurements") or {}
    probes = arm_receipt.get("probes") or {}
    return {
        "H_final": float(m.get("H_bits_per_weight", m.get("H_final", float("nan")))),
        "n_flips": int(m.get("n_flips", 0)),
        "q_changed_count": int(m.get("q_changed_count", 0)),
        "n_applied_drains": int(m.get("n_applied_drains", 0)),
        "lifetime_censored_frac": float(m.get("lifetime_censored_frac", 1.0)),
        "retention_ok": bool(probes.get("retention_ok", False)),
        "acq_delta_count": int(probes.get("acq_delta_count", -10**9)),
    }



def validate_phase0_receipt_for_aggregate(
    phase0_receipt: Mapping[str, Any] | None,
    *,
    phase0_predecessor_receipt: Mapping[str, Any] | None = None,
    censor_clear_max: float = CENSOR_CLEAR_MAX,
    expected_plan_sha256: str = DEFAULT_PLAN_SHA256,
    expected_parent_sha256: str = DEFAULT_PARENT_SHA256,
    expected_authority_dispatch: str = DEFAULT_AUTHORITY_DISPATCH,
    expected_screen_id: str = PHASE0_SCREEN_ID,
    expected_arm_id: str = PHASE0_ARM_ID,
    expected_batch: int = PHASE_BATCH,
    expected_topk: int = PHASE_TOPK,
    allowed_steps: frozenset[int] = PHASE0_ALLOWED_STEPS,
    phase0_steps: int = PHASE0_STEPS,
    phase0b_steps: int = PHASE0B_STEPS_FALLBACK_ONCE,
) -> dict[str, Any]:
    """Formal Phase-0 proof with full provenance + fixed geometry contract.

    Missing/malformed/mismatched → ok=False (caller emits F4/STOP, never authoritative).
    steps==600 requires a hash-bound failed (lcf>=0.50) 150-step predecessor.
    Only after contract passes is lcf<0.50 applied for cleared on the Phase-0 receipt.
    """
    if phase0_receipt is None:
        return {
            "ok": False,
            "phase0_censor_cleared": False,
            "reason": "phase0_proof_missing",
            "lifetime_censored_frac": None,
        }

    def _fail(reason: str, lcf=None, **extra) -> dict[str, Any]:
        out = {
            "ok": False,
            "phase0_censor_cleared": False,
            "reason": reason,
            "lifetime_censored_frac": lcf,
        }
        out.update(extra)
        return out

    def _contract_fields(
        receipt: Mapping[str, Any],
        *,
        expected_steps: int | None,
        label: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return (fail_dict_or_None, extracted)."""
        if str(receipt.get("screen")) != str(expected_screen_id):
            return _fail(f"{label}_screen_mismatch"), {}
        if str(receipt.get("arm")) != str(expected_arm_id):
            return _fail(f"{label}_arm_mismatch"), {}
        if str(receipt.get("plan_sha256")) != str(expected_plan_sha256):
            return _fail(f"{label}_plan_sha256_mismatch"), {}
        if str(receipt.get("authority_dispatch")) != str(expected_authority_dispatch):
            return _fail(f"{label}_authority_dispatch_mismatch"), {}

        banked = receipt.get("banked_sha") or {}
        if not bool(banked.get("match")):
            return _fail(f"{label}_banked_sha_match_false"), {}
        if str(banked.get("before")) != str(expected_parent_sha256):
            return _fail(f"{label}_banked_parent_mismatch"), {}
        if str(banked.get("after")) != str(expected_parent_sha256):
            return _fail(f"{label}_banked_after_mismatch"), {}

        scale = receipt.get("frozen_scale_sha") or {}
        if not bool(scale.get("match")):
            return _fail(f"{label}_frozen_scale_match_false"), {}
        if not scale.get("before"):
            return _fail(f"{label}_frozen_scale_before_missing"), {}

        qsha = receipt.get("q_sha") or {}
        if not qsha.get("before"):
            return _fail(f"{label}_q_sha_before_missing"), {}

        try:
            steps = int(receipt.get("steps", -1))
            batch = int(receipt.get("batch", -1))
            topk = int(receipt.get("topk", -1))
        except (TypeError, ValueError):
            return _fail(f"{label}_geometry_invalid"), {}

        if expected_steps is not None:
            if steps != int(expected_steps):
                return _fail(f"{label}_steps_mismatch"), {}
        elif steps not in set(allowed_steps):
            return _fail(f"{label}_steps_out_of_window"), {}

        if batch != int(expected_batch):
            return _fail(f"{label}_batch_mismatch"), {}
        if topk != int(expected_topk):
            return _fail(f"{label}_topk_mismatch"), {}

        route = receipt.get("route_counters") or {}
        try:
            n_fixed = int(route.get("n_fixed_qscale_forwards", 0))
            n_dyn = int(route.get("n_bitlinear_dynamic_forwards", -1))
        except (TypeError, ValueError):
            return _fail(f"{label}_route_counters_invalid"), {}
        if n_fixed <= 0:
            return _fail(f"{label}_route_counters_n_fixed_nonpositive"), {}
        if n_dyn != 0:
            return _fail(f"{label}_route_counters_dynamic_nonzero"), {}

        m = receipt.get("measurements") or {}
        if "lifetime_censored_frac" not in m:
            return _fail(f"{label}_lifetime_censored_frac_missing"), {}
        try:
            lcf = float(m["lifetime_censored_frac"])
        except (TypeError, ValueError):
            return _fail(f"{label}_lifetime_censored_frac_invalid"), {}

        return None, {
            "steps": steps,
            "batch": batch,
            "topk": topk,
            "lifetime_censored_frac": lcf,
            "frozen_scale_sha_before": str(scale.get("before")),
            "q_sha_before": str(qsha.get("before")),
        }

    fail, fields = _contract_fields(phase0_receipt, expected_steps=None, label="phase0")
    if fail is not None:
        return fail

    steps = int(fields["steps"])
    pred_meta: dict[str, Any] | None = None

    if steps == int(phase0b_steps):
        # Fallback-once: require failed 150-step predecessor with matching state.
        if phase0_predecessor_receipt is None:
            return _fail("phase0_fallback_predecessor_missing")
        pred_fail, pred_fields = _contract_fields(
            phase0_predecessor_receipt,
            expected_steps=int(phase0_steps),
            label="phase0_predecessor",
        )
        if pred_fail is not None:
            # Normalize reason to keep "phase0_predecessor_*" prefix from label.
            return pred_fail
        if pred_fields["frozen_scale_sha_before"] != fields["frozen_scale_sha_before"]:
            return _fail("phase0_fallback_predecessor_scale_mismatch")
        if pred_fields["q_sha_before"] != fields["q_sha_before"]:
            return _fail("phase0_fallback_predecessor_q_mismatch")
        pred_lcf = float(pred_fields["lifetime_censored_frac"])
        if pred_lcf < float(censor_clear_max):
            # Cleared 150 does not authorize escalation to 600.
            return _fail(
                "phase0_fallback_predecessor_cleared",
                lcf=pred_lcf,
            )
        pred_meta = {
            "steps": int(pred_fields["steps"]),
            "batch": int(pred_fields["batch"]),
            "topk": int(pred_fields["topk"]),
            "lifetime_censored_frac": pred_lcf,
            "frozen_scale_sha_before": pred_fields["frozen_scale_sha_before"],
            "q_sha_before": pred_fields["q_sha_before"],
            "failed_censor_guard": True,
        }
    elif steps != int(phase0_steps):
        return _fail("phase0_steps_out_of_window")

    lcf = float(fields["lifetime_censored_frac"])
    cleared = lcf < float(censor_clear_max)
    out = {
        "ok": True,
        "phase0_censor_cleared": cleared,
        "reason": None if cleared else "phase0_censor_uncleared",
        "lifetime_censored_frac": lcf,
        "frozen_scale_sha_before": fields["frozen_scale_sha_before"],
        "q_sha_before": fields["q_sha_before"],
        "steps": steps,
        "batch": int(fields["batch"]),
        "topk": int(fields["topk"]),
    }
    if pred_meta is not None:
        out["phase0_predecessor"] = pred_meta
    return out


class ArmReceiptContractError(ValueError):
    """Shared-held-fixed / provenance mismatch across arm receipts."""


def validate_shared_held_fixed_arm_receipts(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    expected_plan_sha256: str,
    expected_parent_sha256: str,
    expected_authority_dispatch: str = DEFAULT_AUTHORITY_DISPATCH,
    expected_acq_sha: str = ACQUISITION_SELECTION_SHA256,
    expected_id_sha: str = IDENTITY_SELECTION_SHA256,
    expected_batch: int = PHASE_BATCH,
    expected_topk: int = PHASE_TOPK,
    expected_steps: int | None = None,
    allowed_steps: frozenset[int] = PHASE0_ALLOWED_STEPS,
    phase0_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on PLAN_v9 shared-held-fixed contract across arm0..arm3.

    Exact prereg geometry (batch/topk) + optional Phase-0 winning-window steps bind.
    Identical frozen_scale_sha.before + q_sha.before across arms (and Phase-0).
    """
    required = (ARM0, ARM1, ARM2, ARM3)
    for arm in required:
        if arm not in by_arm:
            raise ArmReceiptContractError(f"missing arm receipt for {arm}")

    ref = by_arm[ARM0]
    ref_steps = int(ref.get("steps", -1))
    ref_scale_before = str((ref.get("frozen_scale_sha") or {}).get("before") or "")
    ref_q_before = str((ref.get("q_sha") or {}).get("before") or "")
    if ref_steps not in set(allowed_steps):
        raise ArmReceiptContractError(
            f"control steps {ref_steps} not in PLAN_v9 phase window {sorted(allowed_steps)}"
        )
    if expected_steps is not None and ref_steps != int(expected_steps):
        raise ArmReceiptContractError(
            f"control steps {ref_steps} != Phase-0 winning window {expected_steps}"
        )
    if not ref_scale_before:
        raise ArmReceiptContractError("control missing frozen_scale_sha.before")
    if not ref_q_before:
        raise ArmReceiptContractError("control missing q_sha.before")

    for arm in required:
        r = by_arm[arm]
        if str(r.get("arm")) != arm:
            raise ArmReceiptContractError(
                f"arm label mismatch: key={arm} receipt.arm={r.get('arm')!r}"
            )
        if str(r.get("plan_sha256")) != str(expected_plan_sha256):
            raise ArmReceiptContractError(
                f"{arm} plan_sha256 mismatch: {r.get('plan_sha256')!r}"
            )
        if str(r.get("authority_dispatch")) != str(expected_authority_dispatch):
            raise ArmReceiptContractError(
                f"{arm} authority_dispatch mismatch: {r.get('authority_dispatch')!r}"
            )
        banked = r.get("banked_sha") or {}
        if not bool(banked.get("match")):
            raise ArmReceiptContractError(f"{arm} banked_sha.match is not true")
        if str(banked.get("before")) != str(expected_parent_sha256):
            raise ArmReceiptContractError(
                f"{arm} banked parent mismatch: {banked.get('before')!r}"
            )
        if str(banked.get("after")) != str(expected_parent_sha256):
            raise ArmReceiptContractError(
                f"{arm} banked after mismatch: {banked.get('after')!r}"
            )
        scale = r.get("frozen_scale_sha") or {}
        if not bool(scale.get("match")):
            raise ArmReceiptContractError(f"{arm} frozen_scale_sha.match is not true")
        if str(scale.get("before")) != ref_scale_before:
            raise ArmReceiptContractError(
                f"{arm} frozen_scale_sha.before divergence vs control"
            )
        qsha = r.get("q_sha") or {}
        if str(qsha.get("before")) != ref_q_before:
            raise ArmReceiptContractError(f"{arm} q_sha.before divergence vs control")
        try:
            arm_steps = int(r.get("steps", -1))
            arm_batch = int(r.get("batch", -1))
            arm_topk = int(r.get("topk", -1))
        except (TypeError, ValueError) as e:
            raise ArmReceiptContractError(f"{arm} geometry invalid: {e}") from e
        if arm_steps not in set(allowed_steps):
            raise ArmReceiptContractError(
                f"{arm} steps {arm_steps} not in PLAN_v9 phase window"
            )
        if arm_steps != ref_steps:
            raise ArmReceiptContractError(
                f"{arm} steps mismatch: {arm_steps} != {ref_steps}"
            )
        if expected_steps is not None and arm_steps != int(expected_steps):
            raise ArmReceiptContractError(
                f"{arm} steps {arm_steps} != Phase-0 winning window {expected_steps}"
            )
        if arm_batch != int(expected_batch):
            raise ArmReceiptContractError(
                f"{arm} batch {arm_batch} != prereg PHASE_BATCH {expected_batch}"
            )
        if arm_topk != int(expected_topk):
            raise ArmReceiptContractError(
                f"{arm} topk {arm_topk} != prereg PHASE_TOPK {expected_topk}"
            )
        if str(r.get("screen")) != str(PHASE0_SCREEN_ID):
            raise ArmReceiptContractError(
                f"{arm} screen {r.get('screen')!r} != {PHASE0_SCREEN_ID}"
            )
        if bool(r.get("schema_only", False)):
            raise ArmReceiptContractError(f"{arm} schema_only must be false")
        if bool(r.get("correctness_smoke", False)):
            raise ArmReceiptContractError(f"{arm} correctness_smoke must be false")
        route = r.get("route_counters") or {}
        try:
            n_fixed = int(route.get("n_fixed_qscale_forwards", 0))
            n_dyn = int(route.get("n_bitlinear_dynamic_forwards", -1))
            n_elig = int(route.get("n_eligible_keys", -1))
            n_cred = int(route.get("n_credit_grads_present", -2))
        except (TypeError, ValueError) as e:
            raise ArmReceiptContractError(f"{arm} route_counters invalid: {e}") from e
        if n_fixed <= 0:
            raise ArmReceiptContractError(
                f"{arm} n_fixed_qscale_forwards must be > 0 (got {n_fixed})"
            )
        if n_dyn != 0:
            raise ArmReceiptContractError(
                f"{arm} n_bitlinear_dynamic_forwards must be 0 (got {n_dyn})"
            )
        if n_elig <= 0 or n_cred <= 0 or n_elig != n_cred:
            raise ArmReceiptContractError(
                f"{arm} eligible/credit coverage require "
                f"n_eligible_keys == n_credit_grads_present > 0 "
                f"(got eligible={n_elig} credit={n_cred})"
            )
        probes = r.get("probes") or {}
        if bool(probes.get("skipped", True)):
            raise ArmReceiptContractError(f"{arm} probes.skipped must be false")
        if str(probes.get("acquisition_selection_sha256")) != str(expected_acq_sha):
            raise ArmReceiptContractError(f"{arm} acquisition selection sha mismatch")
        if str(probes.get("identity_selection_sha256")) != str(expected_id_sha):
            raise ArmReceiptContractError(f"{arm} identity selection sha mismatch")
        if int(probes.get("acquisition_n", -1)) != ACQ_N:
            raise ArmReceiptContractError(f"{arm} acquisition_n != {ACQ_N}")
        if int(probes.get("retention_n", -1)) != ACQ_N:
            raise ArmReceiptContractError(f"{arm} retention_n != {ACQ_N}")
        for need in (
            "acq_step0_count",
            "acq_final_count",
            "acq_delta_count",
            "retention_step0_count",
            "retention_final_count",
            "retention_ok",
        ):
            if need not in probes:
                raise ArmReceiptContractError(f"{arm} probes missing {need}")

    if phase0_receipt is not None:
        p0_scale = str((phase0_receipt.get("frozen_scale_sha") or {}).get("before") or "")
        p0_q = str((phase0_receipt.get("q_sha") or {}).get("before") or "")
        if p0_scale != ref_scale_before:
            raise ArmReceiptContractError(
                "phase0 frozen_scale_sha.before divergence vs control"
            )
        if p0_q != ref_q_before:
            raise ArmReceiptContractError("phase0 q_sha.before divergence vs control")

    return {
        "steps": ref_steps,
        "batch": int(expected_batch),
        "topk": int(expected_topk),
        "parent_sha256": expected_parent_sha256,
        "plan_sha256": expected_plan_sha256,
        "authority_dispatch": expected_authority_dispatch,
        "frozen_scale_sha_before": ref_scale_before,
        "q_sha_before": ref_q_before,
        "phase0_winning_window_steps": (
            int(expected_steps) if expected_steps is not None else ref_steps
        ),
    }


def decide_phase0_aggregate_transition(
    p0_val: Mapping[str, Any],
    *,
    phase0_steps: int = PHASE0_STEPS,
    phase0b_steps: int = PHASE0B_STEPS_FALLBACK_ONCE,
) -> dict[str, Any]:
    """PLAN_v9 Phase-0 → Phase-1 state machine (post-validation).

    Returns action in:
      - malformed: ok=False from validator (caller emits non-auth F4)
      - fallback_required: uncleared 150 → non-auth transition (do not classify arms)
      - design_null_censor_unreducible: uncleared 600 w/ failed-150 pred → auth F4, no Phase-1
      - enter_phase1: cleared Phase-0/0b → require arms + classify
    """
    if not bool(p0_val.get("ok")):
        return {
            "action": "malformed",
            "authoritative": False,
            "phase0_censor_cleared": False,
            "stop_reason": str(p0_val.get("reason") or "phase0_proof_invalid"),
            "family": FAMILY_F4,
            "transition": None,
        }
    cleared = bool(p0_val.get("phase0_censor_cleared"))
    steps = int(p0_val.get("steps", -1))
    if cleared:
        return {
            "action": "enter_phase1",
            "authoritative": True,
            "phase0_censor_cleared": True,
            "stop_reason": None,
            "family": None,
            "transition": None,
            "steps": steps,
        }
    if steps == int(phase0_steps):
        return {
            "action": "fallback_required",
            "authoritative": False,
            "phase0_censor_cleared": False,
            "stop_reason": "phase0_censor_uncleared_fallback_required",
            "family": None,
            "transition": "fallback_required",
            "steps": steps,
        }
    if steps == int(phase0b_steps):
        return {
            "action": "design_null_censor_unreducible",
            "authoritative": True,
            "phase0_censor_cleared": False,
            "stop_reason": "design_null_censor_unreducible",
            "family": FAMILY_F4,
            "transition": "design_null_censor_unreducible",
            "steps": steps,
        }
    return {
        "action": "malformed",
        "authoritative": False,
        "phase0_censor_cleared": False,
        "stop_reason": "phase0_steps_out_of_window",
        "family": FAMILY_F4,
        "transition": None,
    }


def build_phase1_terminal_receipt(
    *,
    phase0_censor_cleared: bool,
    control_receipt: Mapping[str, Any] | None,
    arm_receipts: Mapping[str, Mapping[str, Any]] | None,
    plan_sha256: str,
    authority_dispatch: str,
    phase0_proof: Mapping[str, Any] | None = None,
    source_receipt_sha256s: Mapping[str, str] | None = None,
    shared_contract: Mapping[str, Any] | None = None,
    authoritative: bool = True,
    synthetic_phase0_override: bool = False,
    force_null_reason: str | None = None,
    null_family: str | None = FAMILY_F4,
    transition: str | None = None,
    arms_classified: bool = True,
) -> dict[str, Any]:
    """Aggregate arm0..arm3 receipts → classifier → terminal/transition receipt."""
    control_receipt = control_receipt or {}
    arm_receipts = arm_receipts or {}
    if force_null_reason is not None:
        verdict = {
            "eps": EPS,
            "tie_tolerance_bpw": TIE_TOLERANCE_BPW,
            "family": null_family,
            "stop_reason": str(force_null_reason),
            "E": [],
            "S": [],
            "H_progress": {},
            "multi_match": False,
            "arms_classified": False,
        }
        H_control = float(
            (control_receipt.get("measurements") or {}).get(
                "H_bits_per_weight", float("nan")
            )
        )
        metrics = {
            ARM1: arm_metrics_for_classifier(arm_receipts.get(ARM1, {})),
            ARM2: arm_metrics_for_classifier(arm_receipts.get(ARM2, {})),
            ARM3: arm_metrics_for_classifier(arm_receipts.get(ARM3, {})),
        }
    else:
        H_control = float(
            (control_receipt.get("measurements") or {}).get(
                "H_bits_per_weight", float("nan")
            )
        )
        metrics = {
            ARM1: arm_metrics_for_classifier(arm_receipts[ARM1]),
            ARM2: arm_metrics_for_classifier(arm_receipts[ARM2]),
            ARM3: arm_metrics_for_classifier(arm_receipts[ARM3]),
        }
        verdict = classify_forgetting_family_screen(
            phase0_censor_cleared=bool(phase0_censor_cleared),
            H_control_final=H_control,
            arm_metrics=metrics,
        )
        verdict = dict(verdict)
        verdict["arms_classified"] = bool(arms_classified)
    return {
        "screen": "forgetting_mechanism_phase1/v1",
        "plan_sha256": plan_sha256,
        "authority_dispatch": authority_dispatch,
        "authoritative": bool(authoritative) and not bool(synthetic_phase0_override),
        "synthetic_phase0_override": bool(synthetic_phase0_override),
        "phase0_censor_cleared": bool(phase0_censor_cleared),
        "phase0_proof": dict(phase0_proof) if phase0_proof else None,
        "shared_contract": dict(shared_contract) if shared_contract else None,
        "source_receipt_sha256s": (
            dict(source_receipt_sha256s) if source_receipt_sha256s else None
        ),
        "transition": transition,
        "arms_classified": bool(
            False if force_null_reason is not None else arms_classified
        ),
        "H_control_final": H_control,
        "arm_metrics": metrics,
        "classifier": verdict,
        "family": verdict["family"],
        "stop_reason": verdict["stop_reason"],
        "explicit_non_claims": [
            "design-family screen only — no forgetting-law ship",
            "no sub-2 achievability claim",
            "frozen-scale + FixedQScale credit seam bracket",
        ],
    }
