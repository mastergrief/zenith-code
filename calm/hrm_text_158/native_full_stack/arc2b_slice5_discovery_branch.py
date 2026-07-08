"""Arc #2b Slice-5 discovery 2x2 branch classifier (decay-law mechanism question).

Frozen v6 plan (co_lead gate-2 PASS 1783512484577, claude gate-1 freeze
1783512308286, +1 implement 1783526612437). Two science decisions baked in:
  - SYMMETRY WINS: MF-helps and LF-helps are equally valid terminals; B1
    decay-1/1 stability is a liveness prior, NOT live-carrier evidence, so it
    must not privilege LF over MF.
  - TOL for MF AND LF: |gap(1/4) - gap(9/10)| < 0.1*gap(D) => DECAY-DIRECTION-AMBIGUOUS
    unless D already < 0.4 bpw strict (then direction = argmin).

Two booleans partition every (gap_C, gap_D, gap_E) tuple into exactly one
terminal (exhaustive + mutually exclusive by construction):
  MF = gap(1/4) < gap(D)*0.8
  LF = gap(9/10) < gap(D)*0.8
  1. not MF and not LF  -> REPRESENTATION/NEW-MECHANISM (1/2-sweet-spot sub-label
     iff gap(1/2) <= both neighbors; SAME terminal, NOT a peer branch)
  2. MF and not LF      -> MORE-FORGETTING-HELPS (decay-law, toward faster 1/4)
  3. not MF and LF      -> LESS-FORGETTING-HELPS (decay-law, toward slower 9/10)
  4. MF and LF          -> BOTH-IMPROVE (decay-law confirmed; direction = argmin;
     |gap(1/4)-gap(9/10)| < tol => DECAY-DIRECTION-AMBIGUOUS follow-up cell)

Standalone PARTIAL dropped: one material direction IS a finding, not inconclusive.

Operational guard: any live arm ineligible or liveness-fails => no 2x2 terminal
fires; classify operational/inconclusive instead.

Lane-field FAIL-CLOSED: Arm B offline harness fails closed on missing lane
fields (lane_indices/acc_before_lanes/acc_after_lanes/vote_lanes).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Sequence

CLASSIFIER = "ARC2B_SLICE5_DISCOVERY_BRANCH_V1"

ANTI_OVERCLAIM_VERBATIM = (
    "Within the Slice-5 discovery packet scope, the 2x2 decay-law classifier "
    "classifies the live_acc_carrier_bpw gap trend over the 3-point live decay "
    "curve {1/4(C), 1/2(D), 9/10(E)} into one pre-registered branch. FORBIDDEN: "
    "sub-2 readiness, reduction eligibility, bank pin, Fold-3B universalization, "
    "bank mutation, full-stack readiness, H=200/D-terminal verdict, asymptotic "
    "K* at H=100/200, backlog/direction-flip erosion, from-clean-parent "
    "contiguous proof, terminal science claim before registered complete arms "
    "and validation gate."
)

ALLOWED_CLAIM = (
    "Within the Slice-5 discovery packet scope, the 2x2 decay-law classifier "
    "classifies the live_acc_carrier_bpw gap trend over the 3-point live decay "
    "curve into one pre-registered branch."
)

RECEIPT_SCHEMA = "hrm_text_158_arc2b_slice5_discovery_branch_receipt/v1"

# Frozen v6 constants (from arc2b_slice5_in_vivo_branch.py for consistency)
DEFAULT_EFFECTIVE_ACC_BUDGET_BPW = 0.4
DEFAULT_TOLERANCE_BPW = 0.0
DEFAULT_MATERIALITY_FACTOR = 0.8  # gap(arm) < gap(D)*0.8 = "materially improves"
DEFAULT_DIRECTION_TOL_FACTOR = 0.1  # |gap(1/4)-gap(9/10)| < 0.1*gap(D) => AMBIGUOUS

# Decay points on the live curve (all live_acc_carrier_bpw)
DECAY_POINT_C_NUM = 1
DECAY_POINT_C_DEN = 4   # more-forgetting
DECAY_POINT_D_NUM = 1
DECAY_POINT_D_DEN = 2   # control (H=200 law)
DECAY_POINT_E_NUM = 9
DECAY_POINT_E_DEN = 10  # less-forgetting (toward 1/1)

# Arm B is SEPARATE-AXIS (K* saturation, not on live decay-gap curve)
ARM_B_SOURCE_RUN_ID = "2189e72017"
ARM_B_SOURCE_RUN_ROOT = (
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "d_recompute_window_feasibility_seed43_43_2189e72017"
)

# Lane fields required for Arm B offline K* analysis (FAIL-CLOSED on missing)
REQUIRED_LANE_FIELDS: tuple[str, ...] = (
    "lane_indices",
    "acc_before_lanes",
    "acc_after_lanes",
    "vote_lanes",
)

# Arm A static CPU arithmetic floor (fixed-width cannot reach sub-2 ceiling)
ARM_A_W8_BPW = 8  # --dense-accumulator-w8-clip (±127)
ARM_A_W7_BPW = 7  # --dense-accumulator-w7-clip (±63)

CARRIER_BYTE_COMPONENTS: tuple[str, ...] = (
    "events_bytes",
    "backlog_bytes",
    "hot_exact_bytes",
    "metadata_bytes",
)

EVIDENCE_LIVE_DECAY_CURVE = "live_decay_curve"
EVIDENCE_ARM_A_STATIC = "arm_a_static"
EVIDENCE_ARM_B_OFFLINE = "arm_b_offline"

# Terminals that require ALL live arms eligible + no liveness failure
MECHANISM_TERMINALS: frozenset[str] = frozenset(
    {
        "REPRESENTATION_NEW_MECHANISM",
        "MORE_FORGETTING_HELPS",
        "LESS_FORGETTING_HELPS",
        "BOTH_IMPROVE",
        "DECAY_DIRECTION_AMBIGUOUS",
    }
)

# Operational / inconclusive terminals (no mechanism verdict)
OPERATIONAL_TERMINALS: frozenset[str] = frozenset(
    {
        "DISCOVERY_NO_VERDICT_OPERATIONAL",
        "DISCOVERY_INCONCLUSIVE_MISSING_ARM",
        "DISCOVERY_INCONCLUSIVE_LIVENESS_FAILURE",
        "DISCOVERY_INCONCLUSIVE_SOURCE_MISMATCH",
        "DISCOVERY_INCONCLUSIVE_LANE_FIELD_MISSING",
        "DISCOVERY_INCONCLUSIVE_LOG_COVERAGE",
        "DISCOVERY_NO_VERDICT_SCHEMA",
    }
)

REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "schema",
    "task_id",
    "classifier",
    "evidence_source",
    "arm_c_eligible",
    "arm_d_eligible",
    "arm_e_eligible",
    "arm_a_bpw_w8",
    "arm_a_bpw_w7",
    "arm_b_k_star_summary",
    "gap_c_bpw",
    "gap_d_bpw",
    "gap_e_bpw",
    "mf_boolean",
    "lf_boolean",
    "materiality_factor",
    "direction_tol_factor",
    "discovery_branch",
    "discovery_branch_inputs",
    "ready_for_main_science",
    "counts_as_sub2",
    "pre_full_stack_diagnostic",
    "autonomy_rung",
)


class Arc2bSlice5DiscoveryBranch(StrEnum):
    NO_VERDICT_OPERATIONAL = "DISCOVERY_NO_VERDICT_OPERATIONAL"
    INCONCLUSIVE_MISSING_ARM = "DISCOVERY_INCONCLUSIVE_MISSING_ARM"
    INCONCLUSIVE_LIVENESS_FAILURE = "DISCOVERY_INCONCLUSIVE_LIVENESS_FAILURE"
    INCONCLUSIVE_SOURCE_MISMATCH = "DISCOVERY_INCONCLUSIVE_SOURCE_MISMATCH"
    INCONCLUSIVE_LANE_FIELD_MISSING = "DISCOVERY_INCONCLUSIVE_LANE_FIELD_MISSING"
    INCONCLUSIVE_LOG_COVERAGE = "DISCOVERY_INCONCLUSIVE_LOG_COVERAGE"
    NO_VERDICT_SCHEMA = "DISCOVERY_NO_VERDICT_SCHEMA"
    REPRESENTATION_NEW_MECHANISM = "REPRESENTATION_NEW_MECHANISM"
    MORE_FORGETTING_HELPS = "MORE_FORGETTING_HELPS"
    LESS_FORGETTING_HELPS = "LESS_FORGETTING_HELPS"
    BOTH_IMPROVE = "BOTH_IMPROVE"
    DECAY_DIRECTION_AMBIGUOUS = "DECAY_DIRECTION_AMBIGUOUS"


BRANCH_PRECEDENCE: tuple[Arc2bSlice5DiscoveryBranch, ...] = (
    Arc2bSlice5DiscoveryBranch.NO_VERDICT_OPERATIONAL,
    Arc2bSlice5DiscoveryBranch.NO_VERDICT_SCHEMA,
    Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_MISSING_ARM,
    Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_LIVENESS_FAILURE,
    Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_SOURCE_MISMATCH,
    Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_LANE_FIELD_MISSING,
    Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_LOG_COVERAGE,
    Arc2bSlice5DiscoveryBranch.REPRESENTATION_NEW_MECHANISM,
    Arc2bSlice5DiscoveryBranch.MORE_FORGETTING_HELPS,
    Arc2bSlice5DiscoveryBranch.LESS_FORGETTING_HELPS,
    Arc2bSlice5DiscoveryBranch.BOTH_IMPROVE,
    Arc2bSlice5DiscoveryBranch.DECAY_DIRECTION_AMBIGUOUS,
)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def compute_budget_gap_bpw(
    *,
    live_acc_carrier_bpw_max: float,
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
) -> float:
    """Selector: budget_gap_bpw = live_acc_carrier_bpw_max - effective_acc_budget_bpw."""
    return float(live_acc_carrier_bpw_max) - float(effective_acc_budget_bpw)


def arm_eligible(
    *,
    operational_ok: bool,
    live_carrier_bytes_exact: bool,
    resume_generation: int | None,
    liveness_failure: bool = False,
) -> bool:
    """Arm eligible ONLY if ALL: operational_ok, live_carrier_bytes_exact,
    resume_generation=0, no protected-row/liveness fail."""
    if operational_ok is not True:
        return False
    if live_carrier_bytes_exact is not True:
        return False
    if _coerce_int(resume_generation) != 0:
        return False
    if liveness_failure is True:
        return False
    return True


def _materiality_threshold(
    gap_d: float,
    materiality_factor: float = DEFAULT_MATERIALITY_FACTOR,
) -> float:
    """Threshold T = gap(D) * materiality_factor (0.8)."""
    return float(gap_d) * float(materiality_factor)


def _direction_tol(
    gap_d: float,
    direction_tol_factor: float = DEFAULT_DIRECTION_TOL_FACTOR,
) -> float:
    """Tolerance for BOTH-IMPROVE direction tie: 0.1 * gap(D)."""
    return float(gap_d) * float(direction_tol_factor)


def _pick_terminal(
    fired: Sequence[Arc2bSlice5DiscoveryBranch],
) -> Arc2bSlice5DiscoveryBranch:
    if not fired:
        return Arc2bSlice5DiscoveryBranch.NO_VERDICT_OPERATIONAL
    precedence_index = {branch: idx for idx, branch in enumerate(BRANCH_PRECEDENCE)}
    return min(fired, key=lambda branch: precedence_index[branch])


def _check_sweet_spot_sublabel(
    *,
    gap_c: float,
    gap_d: float,
    gap_e: float,
) -> bool:
    """1/2-sweet-spot sub-label: gap(1/2) <= both neighbors.
    SAME terminal as REPRESENTATION_NEW_MECHANISM, NOT a peer branch."""
    return float(gap_d) <= float(gap_c) and float(gap_d) <= float(gap_e)


def classify_arc2b_slice5_discovery_branch(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the 3-point live decay curve into one 2x2 terminal.

    Inputs (REQUIRED):
      arm_c_eligible: bool
      arm_d_eligible: bool
      arm_e_eligible: bool
      gap_c_bpw: float  (budget_gap_bpw for Arm C, decay 1/4)
      gap_d_bpw: float  (budget_gap_bpw for Arm D, decay 1/2 control)
      gap_e_bpw: float  (budget_gap_bpw for Arm E, decay 9/10)
      materiality_factor: float = 0.8
      direction_tol_factor: float = 0.1
      arm_d_bpw_strict: float | None  (live_acc_carrier_bpw_max for D; if < 0.4 strict, argmin)
    """
    fired: list[Arc2bSlice5DiscoveryBranch] = []
    evidence_source = str(inputs.get("evidence_source") or "")

    # Operational guard: any live arm ineligible => no 2x2 terminal
    arm_c_eligible = inputs.get("arm_c_eligible") is True
    arm_d_eligible = inputs.get("arm_d_eligible") is True
    arm_e_eligible = inputs.get("arm_e_eligible") is True

    if not (arm_c_eligible and arm_d_eligible and arm_e_eligible):
        fired.append(Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_MISSING_ARM)
        terminal = _pick_terminal(fired)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "discovery_branch_inputs": dict(inputs),
            "autonomy_rung": "discovery_h25_sentinel",
        }

    # Liveness guard
    if inputs.get("liveness_failure") is True:
        fired.append(Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_LIVENESS_FAILURE)
        terminal = _pick_terminal(fired)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "discovery_branch_inputs": dict(inputs),
            "autonomy_rung": "discovery_h25_sentinel",
        }

    # Schema guard
    if inputs.get("schema_ok") is not True:
        fired.append(Arc2bSlice5DiscoveryBranch.NO_VERDICT_SCHEMA)
        terminal = _pick_terminal(fired)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "discovery_branch_inputs": dict(inputs),
            "autonomy_rung": "discovery_h25_sentinel",
        }

    gap_c = _coerce_float(inputs.get("gap_c_bpw"))
    gap_d = _coerce_float(inputs.get("gap_d_bpw"))
    gap_e = _coerce_float(inputs.get("gap_e_bpw"))

    if gap_c is None or gap_d is None or gap_e is None:
        fired.append(Arc2bSlice5DiscoveryBranch.INCONCLUSIVE_LOG_COVERAGE)
        terminal = _pick_terminal(fired)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "discovery_branch_inputs": dict(inputs),
            "autonomy_rung": "discovery_h25_sentinel",
        }

    materiality_factor = float(
        inputs.get("materiality_factor") or DEFAULT_MATERIALITY_FACTOR
    )
    direction_tol_factor = float(
        inputs.get("direction_tol_factor") or DEFAULT_DIRECTION_TOL_FACTOR
    )

    threshold = _materiality_threshold(gap_d, materiality_factor)
    tol = _direction_tol(gap_d, direction_tol_factor)

    # Two booleans: MF = gap(1/4) < gap(D)*0.8 ; LF = gap(9/10) < gap(D)*0.8
    mf = gap_c < threshold
    lf = gap_e < threshold

    # 2x2 partition (exhaustive + mutually exclusive by construction)
    if not mf and not lf:
        # Cell 1: REPRESENTATION/NEW-MECHANISM
        terminal = Arc2bSlice5DiscoveryBranch.REPRESENTATION_NEW_MECHANISM
        sweet_spot = _check_sweet_spot_sublabel(
            gap_c=gap_c, gap_d=gap_d, gap_e=gap_e
        )
        fired.append(terminal)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [terminal.value],
            "discovery_branch_inputs": dict(inputs),
            "mf_boolean": mf,
            "lf_boolean": lf,
            "materiality_threshold": threshold,
            "direction_tol": tol,
            "sweet_spot_sublabel": sweet_spot,
            "autonomy_rung": "discovery_h25_mechanism",
        }

    if mf and not lf:
        # Cell 2: MORE-FORGETTING-HELPS (toward faster 1/4)
        terminal = Arc2bSlice5DiscoveryBranch.MORE_FORGETTING_HELPS
        fired.append(terminal)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [terminal.value],
            "discovery_branch_inputs": dict(inputs),
            "mf_boolean": mf,
            "lf_boolean": lf,
            "materiality_threshold": threshold,
            "direction_tol": tol,
            "decay_direction": "faster_1_over_4",
            "autonomy_rung": "discovery_h25_mechanism",
        }

    if not mf and lf:
        # Cell 3: LESS-FORGETTING-HELPS (toward slower 9/10)
        terminal = Arc2bSlice5DiscoveryBranch.LESS_FORGETTING_HELPS
        fired.append(terminal)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [terminal.value],
            "discovery_branch_inputs": dict(inputs),
            "mf_boolean": mf,
            "lf_boolean": lf,
            "materiality_threshold": threshold,
            "direction_tol": tol,
            "decay_direction": "slower_9_over_10",
            "autonomy_rung": "discovery_h25_mechanism",
        }

    # Cell 4: MF and LF -> BOTH-IMPROVE
    # direction = argmin(gap(1/4), gap(9/10))
    # |gap(1/4)-gap(9/10)| < tol => DECAY-DIRECTION-AMBIGUOUS
    # UNLESS D already < 0.4 bpw strict (then argmin)
    arm_d_bpw_strict = _coerce_float(inputs.get("arm_d_bpw_strict"))
    d_already_under_budget = (
        arm_d_bpw_strict is not None
        and arm_d_bpw_strict < DEFAULT_EFFECTIVE_ACC_BUDGET_BPW
    )

    direction_diff = abs(gap_c - gap_e)
    if direction_diff < tol and not d_already_under_budget:
        terminal = Arc2bSlice5DiscoveryBranch.DECAY_DIRECTION_AMBIGUOUS
    else:
        terminal = Arc2bSlice5DiscoveryBranch.BOTH_IMPROVE

    fired.append(terminal)
    direction = "argmin" if terminal == Arc2bSlice5DiscoveryBranch.BOTH_IMPROVE else "ambiguous"
    if terminal == Arc2bSlice5DiscoveryBranch.BOTH_IMPROVE:
        direction = "faster_1_over_4" if gap_c <= gap_e else "slower_9_over_10"

    return {
        "classifier": CLASSIFIER,
        "terminal_branch": terminal.value,
        "fired_branches": [terminal.value],
        "discovery_branch_inputs": dict(inputs),
        "mf_boolean": mf,
        "lf_boolean": lf,
        "materiality_threshold": threshold,
        "direction_tol": tol,
        "direction_diff": direction_diff,
        "decay_direction": direction,
        "d_already_under_budget": d_already_under_budget,
        "autonomy_rung": "discovery_h25_mechanism",
    }


def validate_lane_fields(record: Mapping[str, Any]) -> list[str]:
    """FAIL-CLOSED: Arm B offline harness must have all required lane fields."""
    failures: list[str] = []
    for field in REQUIRED_LANE_FIELDS:
        value = record.get(field)
        if value is None:
            failures.append(f"missing_lane_field:{field}")
            continue
        if not isinstance(value, list):
            failures.append(f"lane_field_not_list:{field}")
            continue
        if len(value) == 0:
            failures.append(f"empty_lane_field:{field}")
    return failures


def validate_discovery_receipt_schema(receipt: Mapping[str, Any]) -> list[str]:
    """Validate a discovery branch receipt against the required schema."""
    failures: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        failures.append("schema_mismatch")
    for key in REQUIRED_RECEIPT_FIELDS:
        if key not in receipt:
            failures.append(f"missing:{key}")
    terminal = receipt.get("discovery_branch") or receipt.get("terminal_branch")
    if terminal is not None:
        all_terminals = MECHANISM_TERMINALS | OPERATIONAL_TERMINALS
        if terminal not in all_terminals:
            failures.append(f"invalid_terminal:{terminal}")
    if receipt.get("ready_for_main_science") is not False:
        failures.append("ready_for_main_science_not_false")
    if receipt.get("counts_as_sub2") is not False:
        failures.append("counts_as_sub2_not_false")
    if receipt.get("pre_full_stack_diagnostic") is not True:
        failures.append("pre_full_stack_diagnostic_not_true")
    return failures
